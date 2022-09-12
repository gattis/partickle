const module = globalThis
const { abs, cos, sin, acos, asin, cbrt, sqrt, PI, random, ceil, floor, tan, max, min, log2 } = Math
import './utils.js'


module.f32 = {name:'f32', conv:'Float32', align: 4, size: 4, getset: (off) => ({
    get() { return this.getFloat32(off,true) },
    set(v) { return this.setFloat32(off,v,true) }
})}
module.u32 = {name:'u32', conv:'Uint32', align: 4, size: 4, getset: (off) => ({
    get() { return this.getUint32(off,true) },
    set(v) { return this.setUint32(off,v,true)}
})}
module.u64 = {name:'u64', conv:'Uint32', size: 8, getset: (off) => ({
    get() { return this.getBigUInt64(off,true) },
    set(v) { return this.setBigUInt64(off,v,true) }
})}
module.i32 = {name:'i32', conv:'Int32', align: 4, size: 4, getset: (off) => ({
    get() { return this.getInt32(off,true) },
    set(v) { return this.setInt32(off,v,true)}
})}
module.uatomic = {...u32, name: 'atomic<u32>'}
module.iatomic = {...i32, name: 'atomic<i32>'}



module.GPU = class GPU {

    constructor(canvas) {
        this.canvas = canvas
    }

    
    async init() {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
        const limits = {}, features = ['timestamp-query']
        for (const feat of adapter.features)
            if (feat != 'multi-planar-formats')
                features.push(feat)
        for (const prop in adapter.limits)
            limits[prop] = adapter.limits[prop]
        
        const dev = await adapter.requestDevice({
            requiredFeatures: features,
            requiredLimits: limits
        })
        dev.addEventListener('uncapturederror', ev => {
            if (this.okay) {
                console.error(ev.error.message)
                this.okay = false
            }
        })
        const ctx = this.canvas.getContext('webgpu')
        const fmt = navigator.gpu.getPreferredCanvasFormat()      
        ctx.configure({ device: dev, format: fmt, alphaMode: 'premultiplied' })
        const threads = floor(sqrt(limits.maxComputeWorkgroupsPerDimension))
        const colorAttachment = {
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: { r: 0.13, g: 0.1, b: 0.10, a: 1.0 },
        }
        const depthAttachment = {
            depthClearValue: 1.0,
            depthLoadOp: 'clear',
            depthStoreOp: 'store',
        }
        Object.assign(this, { dev, adapter, ctx, fmt, threads, colorAttachment, depthAttachment, sampleCount: 4, depthFmt: 'depth32float' })
        this.okay = true


        this.resize()

    }

    
    resize() {
        const size = {width: this.canvas.width, height: this.canvas.height}
        this.depthAttachment.view = this.dev.createTexture(
            { size, sampleCount: this.sampleCount, format:this.depthFmt,usage: GPUTextureUsage.RENDER_ATTACHMENT }).createView()
        this.colorAttachment.view = this.dev.createTexture({
            size, sampleCount: this.sampleCount, format: this.fmt, usage: GPUTextureUsage.RENDER_ATTACHMENT }).createView()
    }
  
    write(args) {
        let off = (args.field ? args.field.off : 0) + (args.arrayIdx ? args.arrayIdx * args.data.constructor.stride : 0)
        let size = args.field ? args.field.type.size : args.data.byteLength
        this.dev.queue.writeBuffer(args.buf.buffer, off, args.data.buffer, off + args.data.byteOffset, size)
    }
    
    buf(args) {
        if ('data' in args) [args.type, args.length] = [args.data.constructor, args.data.length]
        args.length = args.length == undefined ? 1 : args.length
        args.size ||= ceil(('data' in args ? args.data.byteLength : args.length * args.type.size) / 4) * 4
        let flags = 0
        args.usage.split('|').forEach(flag => { flags = (flags | GPUBufferUsage[flag]) >>> 0 })
        const mappedAtCreation = 'data' in args
        args.buffer = this.dev.createBuffer({ size: args.size, mappedAtCreation, usage: flags })
        if (mappedAtCreation) {
            new Uint8Array(args.buffer.getMappedRange()).set(new Uint8Array(args.data.buffer))
            args.buffer.unmap()
        }
        args.resource = {buffer: args.buffer}
        return args
    }

    offset(buf,off) {
        const resource = { ...buf.resource, offset: off }
        return { ...buf, resource }
    }

    shader(args) {
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
        args.module = this.dev.createShaderModule({ code })
        args.binds = binds
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
        const buffers = args.vertBufs ||= []
        let { shader, frag, vert, vertBufs, binds, topology, depthWriteEnabled, blend, cullMode } = args
        if (depthWriteEnabled == undefined) depthWriteEnabled = true
        topology ||= 'triangle-list'
        cullMode ||= 'back'
        const visibility = GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX
        const entries = shader.binds.filter(b => binds.includes(b.label)).map(b => (
            { binding:b.idx, ...b.layout,
              visibility: GPUShaderStage.FRAGMENT | (b.label in shader.uniform ? GPUShaderStage.VERTEX : 0) }))
        args.layout = this.dev.createBindGroupLayout({ entries })
        const pipeDesc = {        
            layout: this.dev.createPipelineLayout({ bindGroupLayouts: [ args.layout ] }),
            multisample: { count: this.sampleCount },
            vertex: { module: shader.module, entryPoint:vert, buffers:vertBufs },
            fragment: { module: shader.module , entryPoint:frag, targets: [{ format: this.fmt, blend }]},
            primitive: { topology, cullMode },
            depthStencil: { depthWriteEnabled , depthCompare:'less', format:this.depthFmt },
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
        const texture = this.dev.createTexture({ size, format:this.fmt, sampleCount:1,
                                                 usage: GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_DST })
        bitmaps.forEach((bitmap,z) => {
            let source = { source: bitmap, origin: {x:0, y:0, z:0}, flipY: true }
            let destination = { texture: texture, mipLevel: 0, origin: {x:0, y:0, z:z}, aspect: 'all', colorSpace: 'srgb', premultipliedAlpha: false }
            this.dev.queue.copyExternalImageToTexture(source, destination, [bitmap.width, bitmap.height, 1])
        })
        return { resource: texture.createView({ format: this.fmt, dimension: '2d-array', aspect: 'all', baseMipLevel: 0,
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
        return (encoder) => {
            this.colorAttachment.resolveTarget = this.ctx.getCurrentTexture().createView()
            const pass = encoder.beginRenderPass({
                colorAttachments: [this.colorAttachment],
                depthStencilAttachment: this.depthAttachment
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
        
    clearBuffer(buf) {
        return (encoder) => {
            encoder.clearBuffer(buf.buffer)
        }            
    }

    timestamp(label) {
        return (encoder, labels, querySet) => {
            encoder.writeTimestamp(querySet, labels.length)
            labels.push(label)
        }
    }
    
    encode(cmds) {
        const dev = this.dev
        const querySet = dev.createQuerySet({ type: 'timestamp', count: 64 }) 
        const queryBuf = this.buf({ type: u64, length: 64, usage: 'QUERY_RESOLVE|COPY_SRC' })
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


    render(args) {
        const cmds = this.dev.createCommandEncoder()

        this.dev.queue.submit([cmds.finish()])
    }

    async read(buf) {
        if (this.copyBuf && this.copyBuf.size < buf.size) {
            this.copyBuf.buffer.destroy();
            delete this.copyBuf
        }        
        if (!this.copyBuf)
            this.copyBuf = this.buf({length: buf.size/4, type: u32, usage: 'MAP_READ|COPY_DST' })
        const cmds = this.dev.createCommandEncoder()
        cmds.copyBufferToBuffer(buf.buffer, 0, this.copyBuf.buffer, 0, buf.size)
        this.dev.queue.submit([cmds.finish()])
        await this.copyBuf.buffer.mapAsync(GPUMapMode.READ, 0, buf.size)
        const data = new Uint8Array(this.copyBuf.buffer.getMappedRange(0,buf.size)).slice()
        this.copyBuf.buffer.unmap()
        return data.buffer
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
                for (const [i, { name, type }] of enumerate(cls.fields)) {
                    console.log(type)
                    obj[name] = this[i].toString()
                }
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
            //static toWGSL() {
            //    let fieldWGSL = cls.fields.map(f => `${f.name}: ${f.type.name}`).join(',\n')
            //    return `struct ${cls.name} {\n${fieldWGSL}\n}\n`
        }
        
        cls.stride = roundUp(opt.type.size, opt.type.align)
        cls.size = cls.stride * opt.length
        cls.align = opt.type.align
        cls.isArray = true
        cls.isStruct = true
        for (const [k,v] of Object.entries(opt.members || {}))
            cls.prototype[k] = v
        for (const [k,v] of Object.entries(opt.statics || {}))
            cls[k] = v
        return cls
    }

}


module.i32array = GPU.array({ type: i32 })
module.u32array = GPU.array({ type: u32 })
module.iatomicarray = GPU.array({ type: iatomic })
module.uatomicarray = GPU.array({ type: uatomic })


module.Vec2 = GPU.struct({
    name: 'vec2<f32>',
    fields: [['x', f32], ['y', f32]],
    size: 8, align: 8
})

module.uVec2 = GPU.struct({
    name: 'vec2<u32>',
    fields: [['x', u32], ['y', u32]],
    size: 8, align: 8
})

module.iVec3 = GPU.struct({
    name: 'vec3<i32>',
    fields: [['x', i32], ['y', i32], ['z', i32]],
    size: 12,
    align: 16
})

module.Vec3 = class extends Float32Array {
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
        return Vec3.of(r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta)).normalized()
    }
    static max(a,b) { return Vec3.of(a.x >= b.x ? a.x : b.x, a.y >= b.y ? a.y : b.y, a.z >= b.z ? a.z : b.z) }
    static min(a,b) { return Vec3.of(a.x <= b.x ? a.x : b.x, a.y <= b.y ? a.y : b.y, a.z <= b.z ? a.z : b.z) }
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
    abs() { return Vec3.of(abs(this[0]), abs(this[1]), abs(this[2])) }
    copy() { return Vec3.of(this[0], this[1], this[2]) }
    recip() { return Vec3.of(1/this[0], 1/this[1], 1/this[2]) }
    dist(b) { return sqrt(this.distsq(b)) }
    distsq(b) { return (this[0]-b[0])**2 + (this[1]-b[1])**2 + (this[2]-b[2])**2 }
    dot(b) { return this[0]*b[0] + this[1]*b[1] + this[2]*b[2] }
    cross(b) { return Vec3.of(this[1]*b[2]-this[2]*b[1], this[2]*b[0]-this[0]*b[2], this[0]*b[1]-this[1]*b[0]) }
    addc(b) { return Vec3.of(this[0]+b, this[1]+b, this[2]+b) }
    add(b) { return Vec3.of(this[0]+b[0], this[1]+b[1], this[2]+b[2]) }
    subc(b) { return Vec3.of(this[0]-b, this[1]-b, this[2]-b) }
    sub(b) { return Vec3.of(this[0]-b[0], this[1]-b[1], this[2]-b[2]) }
    mulc(b) { return Vec3.of(this[0]*b, this[1]*b, this[2]*b) }
    mul(b) { return Vec3.of(this[0]*b[0], this[1]*b[1], this[2]*b[2]) }
    divc(b) { return Vec3.of(this[0]/b, this[1]/b, this[2]/b) }
    div(b) { return Vec3.of(this[0]/b[0], this[1]/b[1], this[2]/b[2]) }
    mag() { return sqrt(this[0]**2 + this[1]**2 + this[2]**2) }
    minax() { let ax = this[0] < this[1] ? 0 : 1; return this[ax] < this[2] ? ax : 2 }
    ceil() { return Vec3.of(ceil(this[0]), ceil(this[1]), ceil(this[2])) }
    floor() { return Vec3.of(floor(this[0]), floor(this[1]), floor(this[2])) }
    modc(b) { return Vec3.of(this[0] % b, this[1] % b, this[2] % b) }
    round() { return Vec3.of(round(this[0]), round(this[1]), round(this[2])) }
    normalized() { const m = this.mag(); return m == 0 ? Vec3.of(0) : this.divc(m) }
    maxc() { return max(this[0],this[1],this[2]) }
    minc() { return min(this[0],this[1],this[2]) }
    max(b) { return Vec3.of(max(this[0],b[0]), max(this[1],b[1]), max(this[2],b[2])) }
    min(b) { return Vec3.of(min(this[0],b[0]), min(this[1],b[1]), min(this[2],b[2])) }
    toString() { return '['+[0,1,2].map(i => this[i].toFixed(5).replace(/\.?0+$/g, '').replace(/^-0$/,'0')).join(' ')+']' }

    static getset = (off, type) => ({
        get() { return new Vec3(this.buffer, this.byteOffset + off, type.size) },
        set(v) { return new Int8Array(this.buffer,this.byteOffset+off,type.size).set(new Int8Array(v.buffer,v.byteOffset,type.size)) }   
    })
    
}


module.Vec4 = GPU.struct({
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

module.Mat3 = GPU.array({
    name: 'mat3x3<f32>',
    type: Vec3,
    length: 3,
    align: 16, size: 48,
    statics: {
        of: (arrmat) => {
            const m = Mat3.alloc()
            for (const i of range(3))
                for (const j of range(3))
                    m[i][j] = arrmat[i][j]
            return m
        }
    },
    members: {
        mulc: function(c) {
            const m = Mat3.alloc()
            for (const i of range(3))
                for (const j of range(3))
                    m[i][j] = this[i][j] * c
            return m
        },
        toString: function() {
            return [this[0].toString(), this[1].toString(), this[2].toString()].join('\n')
        }
    }
    
})

// row-major
// left-handed
// row-vector-Matrix product pre-mult v*M
// v*M1*M1*M3 = v*(M1*M2*M3)
module.Mat4 = GPU.array({   
    name: 'mat4x4<f32>',
    type: Vec4,
    length: 4,
    align: 16, size: 64,
    statics: {
        of: (arrmat) => {
            const m = Mat4.alloc()
            for (const i of range(4))
                for (const j of range(4))
                    m[i][j] = arrmat[i][j]
            return m
        },
        I: () => Mat4.of(
            [[1,0,0,0],
             [0,1,0,0],
             [0,0,1,0],
             [0,0,0,1]]
        ),
        translate: (v) => Mat4.of(
            [[  1,   0,   0, 0],
             [  0,   1,   0, 0],
             [  0,   0,   1, 0],
             [v.x, v.y, v.z, 1]]
        ),
        xrot: (a) => Mat4.of(   
            [[1,       0,      0, 0],
             [0,  cos(a), sin(a), 0],
             [0, -sin(a), cos(a), 0],
             [0,       0,      0, 1]]
        ),
        yrot: (b) => Mat4.of(
            [[cos(b), 0, -sin(b), 0],
             [     0, 1,       0, 0],
             [sin(b), 0,  cos(b), 0],
             [     0, 0,       0, 1]]
        ),
        zrot: (g) => Mat4.of(
            [[ cos(g), sin(g), 0, 0],
             [-sin(g), cos(g), 0, 0],
             [      0,      0, 1, 0],
             [      0,      0, 0, 1]]
        ),
        scale: (v) => Mat4.of(
            [[v.x,   0,   0, 0],
             [  0, v.y,   0, 0],
             [  0,   0, v.z, 0],
             [  0,   0,   0, 1]]
        ),     
        look: (pos, dir, up) => {
            const z = dir.normalized().mulc(-1)
            const x = up.cross(z).normalized()
            const y = z.cross(x).normalized()
            return Mat4.of(
                [[        x.x,         y.x,         z.x, 0],
                 [        x.y,         y.y,         z.y, 0],
                 [        x.z,         y.z,         z.z, 0],
                 [-x.dot(pos), -y.dot(pos), -z.dot(pos), 1]]                
            )
        },
        perspective: (deg, aspect, near, far) => {
            const f = 1 / tan(deg * PI / 360)
            const Q = far / (near - far)
            return Mat4.of(
                [[f/aspect, 0,       0,  0],
                 [0,        f,       0,  0],
                 [0,        0,       Q, -1],
                 [0,        0,  Q*near,  0]])
        }
        
    },
    members: {
        transposed: function() {
            const m = Mat4.alloc()
            for (const i of range(4))
                for (const j of range(4))
                    m[i][j] = this[j][i]
            return m
        },
        row: function(i) {
            return this[i]
        },
        col: function(j) {
            return Vec4.of(this[0][j], this[1][j], this[2][j], this[3][j])
        },
        mul: function(b) {
            const c = Mat4.alloc()
            for (const i of range(4))
                for (const j of range(4))
                    c[i][j] = this.row(i).dot(b.col(j))
            return c
        },
        mulc: function(c) {
            const m = Mat4.alloc()
            for (const i of range(4))
                for (const j of range(4))
                    m[i][j] = this[i][j] * c
            return m
        },
        invdet: function() {
            const m0 = this[0], m1 = this[1], m2 = this[2], m3 = this[3]
            const b = [m0[0]*m1[1] - m0[1]*m1[0], m0[0]*m1[2] - m0[2]*m1[0],
                       m0[0]*m1[3] - m0[3]*m1[0], m0[1]*m1[2] - m0[2]*m1[1],
                       m0[1]*m1[3] - m0[3]*m1[1], m0[2]*m1[3] - m0[3]*m1[2],
                       m2[0]*m3[1] - m2[1]*m3[0], m2[0]*m3[2] - m2[2]*m3[0],
                       m2[0]*m3[3] - m2[3]*m3[0], m2[1]*m3[2] - m2[2]*m3[1],
                       m2[1]*m3[3] - m2[3]*m3[1], m2[2]*m3[3] - m2[3]*m3[2]]
            let det = 1 / (b[0]*b[11] - b[1]*b[10] + b[2]*b[9] + b[3]*b[8] - b[4]*b[7] + b[5] * b[6]);
            return [m0,m1,m2,m3,b,det]
        },
        inverse: function() {
            const [m0,m1,m2,m3,b,det] = this.invdet()
            return Mat4.of(
                [[m1.y*b[11]-m1.z*b[10]+m1.w*b[9],m0.z*b[10]-m0.y*b[11]-m0.w*b[9],m3.y*b[5]-m3.z*b[4]+m3.w*b[3],m2.z*b[4]-m2.y*b[5]-m2.w*b[3]],
                 [m1.z*b[8]-m1.x*b[11]-m1.w*b[7],m0.x*b[11]-m0.z*b[8]+m0.w*b[7],m3.z*b[2]-m3.x*b[5]-m3.w*b[1],m2.x*b[5]-m2.z*b[2]+m2.w*b[1]],
                 [m1.x*b[10]-m1.y*b[8]+m1.w*b[6],m0.y*b[8]-m0.x*b[10]-m0.w*b[6],m3.x*b[4]-m3.y*b[2]+m3.w*b[0],m2.y*b[2]-m2.x*b[4]-m2.w*b[0]],
                 [m1.y*b[7]-m1.x*b[9]-m1.z*b[6],m0.x*b[9]-m0.y*b[7]+m0.z*b[6],m3.y*b[1]-m3.x*b[3]-m3.z*b[0],m2.x*b[3]-m2.y*b[1]+m2.z*b[0]]]
            ).mulc(det)
        },
        normal: function() {
            const [m0,m1,m2,m3,b,det] = this.invdet()
            return Mat3.of(
                [[m1[1]*b[11] - m1[2]*b[10] + m1[3]*b[9], m1[2]*b[8] - m1[0]*b[11] - m1[3]*b[7], m1[0]*b[10] - m1[1]*b[8] + m1[3]*b[6]],
                 [m0[2]*b[10] - m0[1]*b[11] - m0[3]*b[9], m0[0]*b[11] - m0[2]*b[8] + m0[3]*b[7], m0[1]*b[8] - m0[0]*b[10] - m0[3]*b[6]],
                 [m3[1]*b[5] - m3[2]*b[4] + m3[3]*b[3], m3[2]*b[2] - m3[0]*b[5] - m3[3]*b[1], m3[0]*b[4] - m3[1]*b[2] + m3[3]*b[0]]]
            ).mulc(det)
        },
        transform: function(v) {
            const out = Vec4.alloc()
            for (const i of range(4))
                out[i] = this.col(i).dot(v)
            return out
        },
        toString: function() {
            return Array.from(range(4)).map(i=>this[i].toString()).join('\n')
        }
    }
})


    
