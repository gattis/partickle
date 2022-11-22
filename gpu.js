const { abs, cos, sin, acos, asin, cbrt, sqrt, PI, random, round, ceil, floor, tan, max, min, log2 } = Math
import * as util from './utils.js'
Object.assign(globalThis, util)

export const f32 = { name:'f32', conv:'Float32', align:4, size: 4, getset: (off) => ({
    get() { return this.getFloat32(off,true) },
    set(v) { return this.setFloat32(off,v,true) }
})}
export const u32 = { name:'u32', conv:'Uint32', align:4, size: 4, getset: (off) => ({
    get() { return this.getUint32(off,true) },
    set(v) { return this.setUint32(off,v,true)}
})}
export const u64 = { name:'u64', conv:'Uint32', size:8, getset: (off) => ({
    get() { return this.getBigUInt64(off,true) },
    set(v) { return this.setBigUInt64(off,v,true) }
})}
export const i32 = { name:'i32', conv:'Int32', align:4, size: 4, getset: (off) => ({
    get() { return this.getInt32(off,true) },
    set(v) { return this.setInt32(off,v,true)}
})}
export const uatomic = {...u32, name: 'atomic<u32>'}
export const iatomic = {...i32, name: 'atomic<i32>'}

const FF = Boolean(globalThis.navigator && navigator.userAgent.match(/(Firefox|Deno)/))
const NODE = globalThis.process && process.release.name == 'node'

export const GPU = class GPU {
    
    async init(width, height, ctx) {
        let promise = navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
        const adapter = await promise
        const limits = {}, features = []
        for (const feature of adapter.features.keys())
            if (feature != 'multi-planar-formats' && feature != 'clear-texture')
               features.push(feature)        
        for (const prop of Object.getOwnPropertyNames(adapter.limits.constructor.prototype)) {
            if (prop == 'maxInterStageShaderVariables') continue
            const val = adapter.limits[prop]
            if (prop != 'constructor' && val != undefined)
                limits[prop] = val
        }
        const desc = { requiredFeatures: features }
        if (!FF) desc.requiredLimits = limits
        const dev = await adapter.requestDevice(desc)
        dev.onuncapturederror = ev => { this.fatal(ev.error) }
        dev.queue.holder = dev
        dev.holder = this
        this.copyBufs = []
        if (features.includes('timestamp-query')) this.ts = true
        
        const threads = floor(sqrt(limits.maxComputeWorkgroupsPerDimension))

        Object.assign(this, { dev, adapter, threads } )
        

    }
    
    configure(ctx, width, height, pref) {
        this.pref = pref
        this.ctx = ctx
        ctx.configure({ device: this.dev, format:pref.format, alphaMode:pref.alpha_mode, colorSpace: 'srgb' })
        const size = {width, height}
        if (this.depthTex) this.depthTex.destroy()
        if (this.colorTex) this.colorTex.destroy()
        const usage = GPUTextureUsage.RENDER_ATTACHMENT
        this.depthTex = this.dev.createTexture({ size, sampleCount: pref.samples, format:pref.depth_fmt, usage })
        this.colorTex = this.dev.createTexture({ size, sampleCount: pref.samples, format:pref.format, usage })
    }

    cleanup() {
        this.dev.destroy()
    }
        
    buf(args) {
        if ('data' in args) [args.type, args.length] = [args.data.constructor, args.data.length]
        args.length = args.length == undefined ? 1 : args.length
        args.size ||= ceil(('data' in args ? args.data.byteLength : args.length * args.type.size) / 4) * 4
        let flags = 0
        args.usage.split('|').forEach(flag => { flags = (flags | GPUBufferUsage[flag]) >>> 0 })
        const mappedAtCreation = 'data' in args
        args.buffer = this.dev.createBuffer({ label: args.label, size: args.size, mappedAtCreation, usage: flags })
        if (mappedAtCreation) {
            new Uint8Array(args.buffer.getMappedRange()).set(new Uint8Array(args.data.buffer))
            args.buffer.unmap()
        }
        args.resource = {buffer: args.buffer, offset: 0, size: args.size}
        return args
    }

    offset(buf,off) {
        const resource = { ...buf.resource, offset: off, size:max(0,buf.size - off) }
        return { ...buf, resource, size: resource.size }
    }

    chop(buf,size) {
        const resource = { ...buf.resource, size }
        return { ...buf, resource, size }
    }

    async read(buf) {
        let copyBuf = this.copyBufs.pop()
        if (!copyBuf || copyBuf.size < buf.size) {
            if (copyBuf) copyBuf.buffer.destroy()
            copyBuf = this.buf({length:buf.size/4, type:u32, usage:'MAP_READ|COPY_DST' })
        }
        const cmds = this.dev.createCommandEncoder()
        cmds.copyBufferToBuffer(buf.buffer, buf.resource.offset, copyBuf.buffer, 0, buf.size)
        this.dev.queue.submit([cmds.finish()])
        await copyBuf.buffer.mapAsync(GPUMapMode.READ, 0, buf.size)
        const data = new Uint8Array(copyBuf.buffer.getMappedRange(0,buf.size)).slice()
        copyBuf.buffer.unmap()
        this.copyBufs.push(copyBuf)
        return data.buffer
    }

    write(buf, data) {
        this.dev.queue.writeBuffer(buf.buffer, buf.resource.offset, data.buffer, data.byteOffset, buf.size)
    }
    
    async shader(args) {
        const { compute, wgsl, defs, storage, uniform, textures, samplers } = args
        const binds = []
        for (const [label,type] of Object.entries(storage||{}))
            binds.push({ label, type, as: compute ? '<storage,read_write>':'<storage>',
                         layout:{ buffer:{ type: compute ? 'storage' : 'read-only-storage' } }, idx:binds.length })
        for (const [label,type] of Object.entries(uniform||{}))
            binds.push({ label, type, as: '<uniform>', layout:{ buffer:{ type:'uniform' } }, idx:binds.length })
        for (const [label,type] of Object.entries(textures||{}))
            binds.push({ label, type, as: '', layout:{ texture:{ viewDimension:'2d-array' }}, idx:binds.length })
        for (const [label,type] of Object.entries(samplers||{}))
            binds.push({ label, type, as: '', layout:{ sampler:{}}, idx:binds.length })
        let code = [
            ...defs.map(struct => struct.toWGSL()),
            ...binds.map(b => `@group(0) @binding(${b.idx}) var${b.as} ${b.label}:${b.type.name};`),
            args.wgsl
        ].join('\n\n')
        if (!compute) code = code.replaceAll('atomic<u32>', 'u32').replaceAll('atomic<i32>','i32')
        if (FF) code = code.replaceAll('const','let')
        args.binds = binds
        args.module = this.dev.createShaderModule({ code })

        try {
            const info = await args.module.compilationInfo()
            for (let msg of info.messages)
                console.warn(msg.message)
        } catch(err) {}
        return args
    }
    
    computePipe(args) {
        const { shader, entryPoint, binds } = args
        const visibility = GPUShaderStage.COMPUTE
        const entries = shader.binds.filter(b => binds.includes(b.label)).map(b => ({ binding:b.idx, visibility, ...b.layout }))
        args.layout = this.dev.createBindGroupLayout({ entries })
        args.pipeline = this.dev.createComputePipeline({
            layout: this.dev.createPipelineLayout({ bindGroupLayouts: [ args.layout ] }),
            compute: { module: shader.module, entryPoint }
        })
        return args
    }

    renderPipe(args) {
        const pref = this.pref
        const buffers = args.vertBufs ||= []
        let { shader, entry, vertBufs, binds, topology, depthWriteEnabled, depthCompare, cullMode, atc,
              color_op, color_src, color_dst, alpha_op, alpha_src, alpha_dst } = args
        color_op ||= pref.color_op; color_src ||= pref.color_src; color_dst ||= pref.color_dst
        alpha_op ||= pref.alpha_op; alpha_src ||= pref.alpha_src; alpha_dst ||= pref.alpha_dst
        
        if (depthWriteEnabled == undefined) depthWriteEnabled = pref.depth_wr
        if (depthCompare == undefined) depthCompare = pref.depth_cmp
        if (atc == undefined) atc = pref.atc
        topology ||= 'triangle-list'
        cullMode ||= pref.cull
        const visibility = GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX
        const entries = shader.binds.filter(b => binds.includes(b.label)).map(b => (
            { binding:b.idx, ...b.layout,
              visibility: GPUShaderStage.FRAGMENT | (b.label in shader.uniform ? GPUShaderStage.VERTEX : 0) }))
        args.layout = this.dev.createBindGroupLayout({ entries })
        const blend = { color: { operation: color_op, srcFactor: color_src, dstFactor: color_dst },
                        alpha: { operation: alpha_op, srcFactor: alpha_src, dstFactor: alpha_dst } }
        const pipeDesc = {
            layout: this.dev.createPipelineLayout({ bindGroupLayouts: [ args.layout ] }),
            multisample: { count: pref.samples, alphaToCoverageEnabled: atc && pref.samples > 1},
            vertex: { module: shader.module, entryPoint:entry+'_vert', buffers:vertBufs },
            fragment: { module: shader.module , entryPoint:entry+'_frag', targets: [{ format: pref.format, blend }]},
            primitive: { topology, cullMode },
            depthStencil: { depthWriteEnabled , depthCompare, format:pref.depth_fmt },
        }
        
        args.pipeline = this.dev.createRenderPipeline(pipeDesc)        
        return args
    }

    texture(bitmaps) {
        const size = [1,1,bitmaps.length || 1]
        bitmaps.forEach(bitmap => {
            size[0] = max(size[0], bitmap.width)
            size[1] = max(size[1], bitmap.height)
        })
        const texture = this.dev.createTexture({
            size, format:this.pref.format, sampleCount:1,  usage: GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_DST })
        bitmaps.forEach((bitmap,z) => {
            let source = { source: bitmap, origin: {x:0, y:0, z:0}, flipY: true }
            let dest = { texture, mipLevel: 0, origin: {x:0, y:0, z:z}, aspect:'all', colorSpace:'srgb', premultipliedAlpha:true }
            this.dev.queue.copyExternalImageToTexture(source, dest, [bitmap.width, bitmap.height, 1])
        })
        return { resource: texture.createView({
            format: this.pref.format, dimension: '2d-array', aspect: 'all', baseMipLevel: 0,
            mipLevelCount: 1, baseArrayLayer:0, arrayLayerCount: bitmaps.length || 1 }) }
    }

    sampler() {
        return { resource: this.dev.createSampler({ magFilter: 'linear', minFilter: 'linear' }) }
    }

    bindGroup(pipe, binds, offsets) {
        const entries = pipe.shader.binds.filter(b => b.label in binds).map(b => ({ binding: b.idx, resource: binds[b.label].resource }))
        return this.dev.createBindGroup({ entries, layout: pipe.layout})
    }
    
    computePass(args) {
        let { pipe, dispatch, binds, offsets } = args
        if (dispatch.length == undefined) dispatch = [dispatch]
        const bg = this.bindGroup(pipe, binds, offsets)
        return (encoder) => { 
            const pass = encoder.beginComputePass()            
            pass.setPipeline(pipe.pipeline)
            pass.setBindGroup(0, bg)
            pass.dispatchWorkgroups(...dispatch)
            pass.end()
        }
    }

    renderPass(draws) {
        const multisamp = this.pref.samples > 1
        return (encoder) => {
            const useStencil = Boolean(this.pref.depth_fmt.match(/stencil/))
            const canvasTex = this.ctx.getCurrentTexture().createView()
            const [view, resolveTarget] = multisamp ? [this.colorTex.createView(), canvasTex] : [canvasTex]
            const pass = encoder.beginRenderPass({
                colorAttachments: [{ 
                    view, resolveTarget,
                    loadOp: 'clear',
                    storeOp: 'store',
                    clearValue: { r: 0, g: 0, b: 0, a: 0 },                    
                }],
                depthStencilAttachment: {
                    depthClearValue: 1.0,
                    depthLoadOp: 'clear',
                    depthStoreOp: 'store',
                    stencilLoadOp: useStencil ? 'clear' : undefined,
                    stencilStoreOp: useStencil ? 'store' : undefined,                    
                    view: this.depthTex.createView()
                }
            })
            for (const draw of draws)
                draw(pass)
            pass.end()
        }
    }

    draw(args) {
        let { pipe, dispatch, binds } = args
        if (dispatch.length == undefined) dispatch = [dispatch]
        const bg = this.bindGroup(pipe, binds)
        return (pass) => {
            pass.setPipeline(pipe.pipeline)
            pass.setBindGroup(0, bg)
            for (const i of range(pipe.vertBufs.length))
                pass.setVertexBuffer(i, pipe.vertBufs[i].buf.buffer)
            pass.draw(...dispatch)
        }
    }

    drawIndexed(args) {
        let { pipe, dispatch, binds } = args
        if (dispatch.length == undefined) dispatch = [dispatch]
        const bg = this.bindGroup(pipe, binds)
        return (pass) => {
            pass.setPipeline(pipe.pipeline)
            pass.setBindGroup(0, bg)
            for (const i of range(pipe.vertBufs.length))
                pass.setVertexBuffer(i, pipe.vertBufs[i].buf.buffer)
            pass.setIndexBuffer(pipe.indexBuf.buf.buffer, pipe.indexBuf.indexFormat)
            pass.drawIndexed(...dispatch)
        }
    }
        
    clearBuffer(buf) {
        return (encoder) => {
            encoder.clearBuffer(buf.buffer, 0, buf.size)
        }            
    }

    timestamp(label) {
        if (!this.ts) return ()=>{}
        return (encoder, labels, querySet) => {
            encoder.writeTimestamp(querySet, labels.length)
            labels.push(label)
        }
    }

    encode(cmds) {
        const dev = this.dev
        const querySet = this.ts ? dev.createQuerySet({ type: 'timestamp', count: 64 }) : null
        let queryBuf
        queryBuf = this.buf({ label:'query', type: u64, length: 64, usage: 'COPY_SRC'+(FF?'':'|QUERY_RESOLVE')})
        return {
            stampLabels: [],
            stampBuf: queryBuf,
            execute() {
                let encoder = dev.createCommandEncoder()
                const labels = []
                for (const cmd of cmds)
                    cmd(encoder, labels, querySet)
                if (labels.length > 0)
                    encoder.resolveQuerySet(querySet, 0, labels.length, queryBuf.buffer, 0)
                dev.queue.submit([encoder.finish()])
                this.stampLabels = labels
            }
        }
    }

    static struct(opt) {
        const cls = class extends DataView {
            static name = opt.name
            static alloc() {
                return new this(new ArrayBuffer(this.size))
            }
            static of(...args) {
                while (args.length < cls.fields.length) args.push(args.at(-1))
                const inst = this.alloc()
                for (let i = 0; i < args.length; i++)
                    inst[i] = args[i]
                return inst
            }
            static toWGSL() {
                let fieldWGSL = cls.fields.map(f => `${f.name}: ${f.type.name}`).join(',\n')
                return `struct ${cls.name} {\n${fieldWGSL}\n}\n`
            }
            get table() {
                const obj = {}
                for (const [i, { name, type }] of enumerate(cls.fields))
                    obj[name] = this[i].toString()
                return obj
            }
            [Symbol.iterator]() {
                let idx = 0;
                return { next: () => idx < cls.fields.length ? { value: this[idx++], done: false } : { done: true } }
            }
            static getset = (off,type) => ({
                get() { return new type(this.buffer, this.byteOffset + off, type.size) },
                set(v) { return new Int8Array(this.buffer,this.byteOffset+off,type.size).set(new Int8Array(v.buffer,v.byteOffset,type.size)) }   
            })
        }
        cls.fields = []
        let align = 0
        let last = {off: 0, type: {size: 0}}
        opt.fields.forEach(([name,type], idx) => {
            const off = roundUp(last.off + last.type.size, type.align)
            last  = {name, type, off}
            cls.fields.push(last)
            cls[name] = last
            Object.defineProperty(cls.prototype, name, type.getset(off,type))
            Object.defineProperty(cls.prototype, idx, type.getset(off,type))
            align = max(align, type.align)
        })
        cls.align = align
        if (opt.align) cls.align = opt.align
        cls.size = roundUp(last.off + last.type.size, align)
        cls.isStruct = true
        for (const [k,v] of Object.entries(opt.members || {}))
            cls.prototype[k] = v
        for (const [k,v] of Object.entries(opt.statics || {}))
            cls[k] = v
        return cls        
    }


    static array(opt) {    
        const cls = class extends DataView {
            static name = opt.name ? opt.name : `array<${opt.type.name}${opt.length?','+opt.length:''}>`
            constructor(...args) {
                super(...args)
                this.type = opt.type
                this.stride = cls.stride
                this.length = this.byteLength / this.stride
                return new Proxy(this,this)
            }
            static alloc(length) {
                const bufSz = length == undefined ? this.size : length * this.stride
                return new this(new ArrayBuffer(bufSz))
            }
            static of(arr) {
                const sarr = this.alloc(arr.length)
                for (let i = 0; i < arr.length; i++)
                    sarr[i] = arr[i]
                return sarr
            }
            get(o,k) {
                if (k == Symbol.iterator || isNaN(k)) return Reflect.get(o,k)
                if (o.type.isStruct) 
                    return new o.type(o.buffer, o.byteOffset + k*o.stride, o.type.size)
                return this[`get${o.type.conv}`](k*o.stride, true)
            }           
            set(o,k,v) {
                if (isNaN(k)) return Reflect.set(o,k,v)
                if (o.type.isStruct) {
                    let src = new Uint8Array(v.buffer, v.byteOffset, o.type.size)
                    let dst = new Uint8Array(o.buffer, o.byteOffset + k*o.stride);
                    dst.set(src)
                    return true
                }
                this[`set${o.type.conv}`](k*o.stride, v, true)
                return true
            }
            apply(o, t, args) {
                return o.lambda(...args)
            }
            subarray(first,last) {
                return this.constructor(this.buffer, this.byteOffset + first * this.stride, this.byteOffset + (last-first)*this.stride)
            }
            [Symbol.iterator]() {
                let idx = 0;
                return {
                    next: () => {
                        if (idx < this.length)
                            return { value: this[idx++], done: false }
                        else
                            return { done: true }
                    }
                }
            }
            static getset = (off,type) => ({
                get() { return new type(this.buffer, this.byteOffset + off, type.size) },
                set(v) { return new Int8Array(this.buffer,this.byteOffset+off,type.size).set(new Int8Array(v.buffer,v.byteOffset,type.size)) }   
            })
        }
        
        cls.stride = roundUp(opt.type.size, opt.type.align)
        if (opt.length != undefined) cls.size = cls.stride * opt.length
        cls.align = opt.type.align
        cls.isArray = true
        cls.isStruct = true
        for (const [k,v] of Object.entries(opt.members || {}))
            cls.prototype[k] = v
        for (const [k,v] of Object.entries(opt.statics || {}))
            cls[k] = v
        return cls
    }

    fatal(err) {
        if (err == null) return
        console.error(err.message);
        debugger;
        //this.cleanup()
    }
}

export const i32array = GPU.array({ type: i32 })
export const u32array = GPU.array({ type: u32 })
export const f32array = GPU.array({ type: f32 })
export const iatomicarray = GPU.array({ type: iatomic })
export const uatomicarray = GPU.array({ type: uatomic })

export const V2 = GPU.struct({
    name: 'vec2<f32>',
    fields: [['x', f32], ['y', f32]],
    size: 8, align: 8,
    members: {
        toString: function() {
            return '['+[0,1].map(i => this[i].toFixed(5).replace(/\.?0+$/g, '').replace(/^-0$/,'0')).join(' ')+']'
        }
    }
})

export const v2 = (...args) => V2.of(...args)

export const V2U = GPU.struct({
    name: 'vec2<u32>',
    fields: [['x', u32], ['y', u32]],
    size: 8, align: 8
})

export const v2u = (...args) => V2U.of(...args)

export const V3U = GPU.struct({
    name: 'vec3<i32>',
    fields: [['x', i32], ['y', i32], ['z', i32]],
    size: 12,
    align: 16
})

export const V3 = class extends Float32Array {
    static name = 'vec3<f32>'
    static isStruct = true
    static align = 16
    static size = 12

    constructor(buffer, byteOffset, byteLength) {
        super(buffer, byteOffset, byteLength / 4)
    }

    static rand() {
        const phi = 2 * PI * rand.f(0,1)
        const theta = acos(rand.f(-1,1))
        const r = cbrt(rand.f(0,1))
        return v3(r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta)).normalized()
    }
    static max(a,b) { return v3(a.x >= b.x ? a.x : b.x, a.y >= b.y ? a.y : b.y, a.z >= b.z ? a.z : b.z) }
    static min(a,b) { return v3(a.x <= b.x ? a.x : b.x, a.y <= b.y ? a.y : b.y, a.z <= b.z ? a.z : b.z) }
    static alloc() { return new this(3) }
    static of(x,y,z) {
        if (y == undefined) y = z = x
        return new this([x, y, z])
    }
    get x() { return this[0] }
    get y() { return this[1] }
    get z() { return this[2] }
    set x(v) { return this[0] = v }
    set y(v) { return this[1] = v }
    set z(v) { return this[2] = v }
    abs() { return v3(abs(this[0]), abs(this[1]), abs(this[2])) }
    copy() { return v3(this[0], this[1], this[2]) }
    recip() { return v3(1/this[0], 1/this[1], 1/this[2]) }
    dist(b) { return sqrt(this.distsq(b)) }
    distsq(b) { return (this[0]-b[0])**2 + (this[1]-b[1])**2 + (this[2]-b[2])**2 }
    dot(b) { return this[0]*b[0] + this[1]*b[1] + this[2]*b[2] }
    cross(b) { return v3(this[1]*b[2]-this[2]*b[1], this[2]*b[0]-this[0]*b[2], this[0]*b[1]-this[1]*b[0]) }
    addc(b) { return v3(this[0]+b, this[1]+b, this[2]+b) }
    add(b) { return v3(this[0]+b[0], this[1]+b[1], this[2]+b[2]) }
    subc(b) { return v3(this[0]-b, this[1]-b, this[2]-b) }
    sub(b) { return v3(this[0]-b[0], this[1]-b[1], this[2]-b[2]) }
    mulc(b) { return v3(this[0]*b, this[1]*b, this[2]*b) }
    mul(b) { return v3(this[0]*b[0], this[1]*b[1], this[2]*b[2]) }
    divc(b) { return v3(this[0]/b, this[1]/b, this[2]/b) }
    div(b) { return v3(this[0]/b[0], this[1]/b[1], this[2]/b[2]) }
    mulm(m) { 
        return v3(this[0]*m[0][0]+ this[1]*m[1][0] + this[2]*m[2][0],
                  this[0]*m[0][1]+ this[1]*m[1][1] + this[2]*m[2][1],
                  this[0]*m[0][2]+ this[1]*m[1][2] + this[2]*m[2][2])
    }
    sum() { return this[0] + this[1] + this[2] }
    mag() { return sqrt(this[0]**2 + this[1]**2 + this[2]**2) }
    majorAxis() { let ax = abs(this[0]) > abs(this[1]) ? 0 : 1; return abs(this[ax]) > abs(this[2]) ? ax : 2 }
    round() { return [round(this[0]), round(this[1]), round(this[2])] }
    toarray() { return [this[0], this[1], this[2]] }
    ceil() { return v3(ceil(this[0]), ceil(this[1]), ceil(this[2])) }
    floor() { return v3(floor(this[0]), floor(this[1]), floor(this[2])) }
    modc(b) { return v3(this[0] % b, this[1] % b, this[2] % b) }
    round() { return v3(round(this[0]), round(this[1]), round(this[2])) }
    normalized() { const m = this.mag(); return m == 0 ? v3(0) : this.divc(m) }
    maxc() { return max(this[0],this[1],this[2]) }
    minc() { return min(this[0],this[1],this[2]) }
    max(b) { return v3(max(this[0],b[0]), max(this[1],b[1]), max(this[2],b[2])) }
    min(b) { return v3(min(this[0],b[0]), min(this[1],b[1]), min(this[2],b[2])) }
    toString() { return 'v3('+[0,1,2].map(i => this[i].toFixed(5).replace(/\.?0+$/g, '').replace(/^-0$/,'0')).join(',')+')' }

    static getset = (off, type) => ({
        get() { return new V3(this.buffer, this.byteOffset + off, type.size) },
        set(v) { return new Int8Array(this.buffer,this.byteOffset+off,type.size).set(new Int8Array(v.buffer,v.byteOffset,type.size)) }   
    })
    
}

export const v3 = (...args) => V3.of(...args)

export const V4 = GPU.struct({
    name: 'vec4<f32>',
    fields: [['x', f32], ['y', f32], ['z', f32], ['w', f32]],
    size: 16, align: 16,
    members: {
        dot: function(b) { return this.x*b.x + this.y*b.y + this.z*b.z + this.w*b.w },
        toString: function() {
            return '['+[0,1,2,3].map(i => this[i].toFixed(5).replace(/\.?0+$/g, '').replace(/^-0$/,'0')).join(' ')+']'
        }
    }
})

export const v4 = (...args) => V4.of(...args)

export const M3 = GPU.array({
    name: 'mat3x3<f32>',
    type: V3,
    length: 3,
    align: 16, size: 48,
    statics: {
        of: (arrmat) => {
            const m = M3.alloc()
            for (const i of range(3))
                for (const j of range(3))
                    m[i][j] = arrmat[i][j]
            return m
        }
    },
    members: {
        col: function (j) {
            return v3(this[0][j], this[1][j], this[2][j])
        },
        mulc: function(c) {
            const m = M3.alloc()
            for (const i of range(3))
                for (const j of range(3))
                    m[i][j] = this[i][j] * c
            return m
        },
        mul: function (b) {
            const c = M3.alloc()
            for (const i of range(3))
                for (const j of range(3))
                    c[i][j] = this[i].dot(b.col(j))
            return c
        },
        toString: function() {
            return [this[0].toString(), this[1].toString(), this[2].toString()].join('\n')
        },
        inverse: function() {
            const [[m00,m01,m02],[m10,m11,m12],[m20,m21,m22]] = this
            const n = m3([[m22*m11 - m12*m21, -m22*m01 + m02*m21, m12*m01 - m02*m11], 
                          [-m22*m10 + m12*m20, m22*m00 - m02*m20, -m12*m00 + m02*m10],
                          [m21*m10 - m11*m20, -m21*m00 + m01*m20, m11*m00 - m01*m10]])
            return n.mulc(1 / (n[0][0]*m00+ n[1][0]*m01 + n[2][0]*m02));
        }
    }
    
})

export const m3 = (...args) => M3.of(...args)

// row-major
// left-handed
// row-vector-Matrix product pre-mult v*M
// v*M1*M1*M3 = v*(M1*M2*M3)
export const M4 = GPU.array({
    name: 'mat4x4<f32>',
    type: V4,
    length: 4,
    align: 16, size: 64,
    statics: {
        of: (arrmat) => {
            const m = M4.alloc()
            for (const i of range(4))
                for (const j of range(4))
                    m[i][j] = arrmat[i][j]
            return m
        },
        I: () => m4(
            [[1,0,0,0],
             [0,1,0,0],
             [0,0,1,0],
             [0,0,0,1]]
        ),
        translate: (v) => m4(
            [[  1,   0,   0, 0],
             [  0,   1,   0, 0],
             [  0,   0,   1, 0],
             [v.x, v.y, v.z, 1]]
        ),
        xrot: (a) => m4(   
            [[1,       0,      0, 0],
             [0,  cos(a), sin(a), 0],
             [0, -sin(a), cos(a), 0],
             [0,       0,      0, 1]]
        ),
        yrot: (b) => m4(
            [[cos(b), 0, -sin(b), 0],
             [     0, 1,       0, 0],
             [sin(b), 0,  cos(b), 0],
             [     0, 0,       0, 1]]
        ),
        zrot: (g) => m4(
            [[ cos(g), sin(g), 0, 0],
             [-sin(g), cos(g), 0, 0],
             [      0,      0, 1, 0],
             [      0,      0, 0, 1]]
        ),
        scale: (v) => m4(
            [[v.x,   0,   0, 0],
             [  0, v.y,   0, 0],
             [  0,   0, v.z, 0],
             [  0,   0,   0, 1]]
        ),     
        look: (pos, dir, up) => {
            const z = dir.normalized().mulc(-1)
            const x = up.cross(z).normalized()
            const y = z.cross(x).normalized()
            return m4(
                [[        x.x,         y.x,         z.x, 0],
                 [        x.y,         y.y,         z.y, 0],
                 [        x.z,         y.z,         z.z, 0],
                 [-x.dot(pos), -y.dot(pos), -z.dot(pos), 1]]                
            )
        },
        perspective: (deg, aspect, near, far) => {
            const f = 1 / tan(deg * PI / 360)
            const Q = far / (near - far)
            return m4(
                [[f/aspect, 0,       0,  0],
                 [0,        f,       0,  0],
                 [0,        0,       Q, -1],
                 [0,        0,  Q*near,  0]])
        }
        
    },
    members: {
        transposed: function() {
            const m = M4.alloc()
            for (const i of range(4))
                for (const j of range(4))
                    m[i][j] = this[j][i]
            return m
        },
        row: function(i) {
            return this[i]
        },
        col: function(j) {
            return v4(this[0][j], this[1][j], this[2][j], this[3][j])
        },
        mul: function(b) {
            const c = M4.alloc()
            for (const i of range(4))
                for (const j of range(4))
                    c[i][j] = this.row(i).dot(b.col(j))
            return c
        },
        mulc: function(c) {
            const m = M4.alloc()
            for (const i of range(4))
                for (const j of range(4))
                    m[i][j] = this[i][j] * c
            return m
        },
        invdet: function() {
            const u = this[0], v = this[1], w = this[2], t = this[3]
            const b = [u[0]*v[1] - u[1]*v[0], u[0]*v[2] - u[2]*v[0],
                       u[0]*v[3] - u[3]*v[0], u[1]*v[2] - u[2]*v[1],
                       u[1]*v[3] - u[3]*v[1], u[2]*v[3] - u[3]*v[2],
                       w[0]*t[1] - w[1]*t[0], w[0]*t[2] - w[2]*t[0],
                       w[0]*t[3] - w[3]*t[0], w[1]*t[2] - w[2]*t[1],
                       w[1]*t[3] - w[3]*t[1], w[2]*t[3] - w[3]*t[2]]
            let det = 1 / (b[0]*b[11] - b[1]*b[10] + b[2]*b[9] + b[3]*b[8] - b[4]*b[7] + b[5] * b[6]);
            return [u,v,w,t,b,det]
        },
        inverse: function() {
            const [u,v,w,t,b,det] = this.invdet()
            return m4(
                [[v.y*b[11]-v.z*b[10]+v.w*b[9],u.z*b[10]-u.y*b[11]-u.w*b[9],t.y*b[5]-t.z*b[4]+t.w*b[3],w.z*b[4]-w.y*b[5]-w.w*b[3]],
                 [v.z*b[8]-v.x*b[11]-v.w*b[7],u.x*b[11]-u.z*b[8]+u.w*b[7],t.z*b[2]-t.x*b[5]-t.w*b[1],w.x*b[5]-w.z*b[2]+w.w*b[1]],
                 [v.x*b[10]-v.y*b[8]+v.w*b[6],u.y*b[8]-u.x*b[10]-u.w*b[6],t.x*b[4]-t.y*b[2]+t.w*b[0],w.y*b[2]-w.x*b[4]-w.w*b[0]],
                 [v.y*b[7]-v.x*b[9]-v.z*b[6],u.x*b[9]-u.y*b[7]+u.z*b[6],t.y*b[1]-t.x*b[3]-t.z*b[0],w.x*b[3]-w.y*b[1]+w.z*b[0]]]
            ).mulc(det)
        },
        normal: function() {
            const [u,v,w,t,b,det] = this.invdet()
            return m3(
                [[v[1]*b[11] - v[2]*b[10] + v[3]*b[9], v[2]*b[8] - v[0]*b[11] - v[3]*b[7], v[0]*b[10] - v[1]*b[8] + v[3]*b[6]],
                 [u[2]*b[10] - u[1]*b[11] - u[3]*b[9], u[0]*b[11] - u[2]*b[8] + u[3]*b[7], u[1]*b[8] - u[0]*b[10] - u[3]*b[6]],
                 [t[1]*b[5] - t[2]*b[4] + t[3]*b[3], t[2]*b[2] - t[0]*b[5] - t[3]*b[1], t[0]*b[4] - t[1]*b[2] + t[3]*b[0]]]
            ).mulc(det)
        },
        transform: function(v) {
            const out = V4.alloc()
            for (const i of range(4))
                out[i] = this.col(i).dot(v)
            return out
        },
        toString: function() {
            return Array.from(range(4)).map(i=>this[i].toString()).join('\n')
        }
    }
})

export const m4 = (...args) => M4.of(...args)

for (const meth of Object.getOwnPropertyNames(GPUDevice.prototype)) {
    if (!(meth == 'destroy' || meth.startsWith('create')) || meth == 'createCommandEncoder') continue;
    hijack(GPUDevice, meth, (real, obj, args) => {
        obj.pushErrorScope('validation')
        const retval = real.apply(obj, args)
        if (meth != 'destroy') obj.popErrorScope().then(err => obj.holder.fatal(err))
        return retval
    })
}

hijack(GPUQueue, 'submit', (real, obj, args) => {
    obj.holder.pushErrorScope('validation')
    real.apply(obj, args)
    obj.holder.popErrorScope().then(err => obj.holder.holder.fatal(err))
})
