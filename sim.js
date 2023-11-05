import * as util from './utils.js'
import * as gpu from './gpu.js'
import * as geo from './geometry.js'
Object.assign(globalThis, util, gpu, geo)

const MAXEDGES = 18
const MAXRING = 48
const CELLSIZE = 8
const CELLDIM = 256
const SHADOWRES = 4096

const particleColors = [v4(.3,.6,.8,1), v4(.99,.44,.57, 1), v4(.9, .48, .48, 1)]

const MESH_DEFAULTS = {
    name:'default', bitmapId:-1, color:[1,1,1,1], offset:[0,0,0], rotation:[0,0,0],
    scale:[1,1,1], volstiff:1, shearstiff:1, friction:1, collision:1, fluid:0, w:1,
}

export const phys = new Preferences('phys')
phys.addBool('paused', false)
phys.addNum('r', 1.0, 0.1, 10, 0.1)
phys.addNum('frameratio', 3, 1, 20, 1)
phys.addNum('a', -9.8, -20, 20, 0.1)
phys.addNum('volstiff', .5, 0, 2, 0.001)
phys.addNum('shearstiff', .5, 0, 1, 0.001)
phys.addNum('damp', 0.5, 0, 1, .1)
phys.addNum('friction', 1, 0, 1, .01)
phys.addNum('collision', 1, 0, 1, .01)
phys.addNum('xmin', -1.5, -25, 0, 0.01)
phys.addNum('xmax', 1.5, .01, 25, 0.01)
phys.addNum('ymin', -1.5, -25, 0, 0.01)
phys.addNum('ymax', 1.5, .01, 25, 0.01)
phys.addNum('zmin', 0, -25, 0, 0.01)
phys.addNum('zmax', 3, 0.01, 50, 0.01)




export const render = new Preferences('render')
render.addBool('particles', true)
render.addBool('walls', true)
render.addBool('normals', false)
render.addBool('shadows', true)
render.addBool('lights', true)
render.addBool('velocity', false)
render.addBool('surfaces', true)

render.addChoice('alpha_mode', 'premultiplied', ['opaque','premultiplied'])
render.addChoice('format', 'rgba16float', ['rgba8unorm','bgra8unorm','rgba16float'])
render.addChoice('depth_fmt', 'depth32float', ['depth24plus','depth32float'])
render.addNum('samples', 4, 1, 4, 3)
render.addNum('fov', 60, 1, 150, 1)



render.addVector('cam_x', v3(0, -2, 2))
render.addNum('cam_ud', 0)
render.addNum('cam_lr', 0)
render.hidden['cam_x'] = true
render.hidden['cam_ud'] = true
render.hidden['cam_lr'] = true


const clock = () => performance.now()/1000

export const Mesh = GPU.struct({
    name:'Mesh',
    fields:[
        ['ci', V3],
        ['vi', V3],
        ['wi', V3],
        ['pi', u32],
        ['pf', u32],
        ['tex', i32],
        ['color', V4],
        ['pcolor', V4],
        ['volstiff', f32],
        ['shearstiff', f32],
        ['fluid', i32],
        ['padding', GPU.array({ type:u32, length:36 })]
    ]
})

export const Particle = GPU.struct({
    name:'Particle',
    fields:[
        ['x', V3],
        ['w', f32],
        ['v', V3],
        ['friction',f32],
        ['x0', V3],
        ['collision', f32],       
        ['dx', V3],
        ['hash',i32],
        ['dv',V3],
    ]
})

export const ParticleVert = GPU.struct({
    name:'ParticleVert',
    fields:[
        ['norm', V3],
        ['mesh',u32],
        ['nedges', u32],
        ['edges', GPU.array({ type:u32, length:MAXEDGES })],
    ]
})
    
export const Eye = GPU.struct({
    nane:'Eye',
    fields:[
        ['viewproj',M4],
        ['x', V3],
        ['dir', V3],
    ]
})

export const Light = GPU.struct({
    name:'Light',
    fields:[
        ['viewproj',M4],
        ['x', V3],
        ['dir', V3],
        ['color', V3],
        ['power', f32],
        ['shadow', i32],
    ]
})

export const Uniforms = GPU.struct({
    name:'Uniforms',
    fields:[
        ['cam_x',V3],
        ['t', f32],
        ['r', f32],
        ['seed',u32],
        ['selection', i32],
        ['grab_query', i32],
        ['grab_pid', i32],
        ['grab_ray', V3],
        ['grab_x', V3],
        ['spacemin',V3],
        ['spacemax',V3],
        ['damp',f32],
        ['a',V3],
        ['friction',f32],
        ['collision',f32],
        ['volstiff',f32],
        ['shearstiff',f32],
    ]
})

export const TriVert = GPU.struct({
    name:'TriVert',
    fields:[
        ['x', V3],
        ['pidx', u32],
        ['norm', V3],
        ['mesh', u32],
        ['uv', V2],
    ]
})

export const Triangle = GPU.struct({
    name:'Triangle',
    fields:[
        ['v0', TriVert],
        ['v1', TriVert],
        ['v2', TriVert],
    ]
})

export const GrabHit = GPU.struct({
    name:'GrabHit',
    fields:[
        ['pid', u32],
        ['x', V3]
    ]
})

export const GrabHits = GPU.struct({
    name:'GrabHits',
    fields:[
        ['list', GPU.array({ type:GrabHit, length:8192 })],
        ['len', iatomic]
    ]
})

export const Cell = GPU.array({ type:iatomic, length: CELLSIZE })

export const Cluster = GPU.struct({
    name:'Cluster',
    fields:[
        ['c0', V3],
        ['n', u32],
        ['s', f32],
        ['mesh', u32],
        ['pids', GPU.array({ type:u32, length:1024 })],
        ['qinv', M3],
    ]
})

const Meshes = GPU.array({ type:Mesh })
const Particles = GPU.array({ type:Particle })
const ParticleVerts = GPU.array({ type:ParticleVert })
const Triangles = GPU.array({ type:Triangle })
const Cells = GPU.array({ type:Cell })
const Clusters = GPU.array({ type:Cluster })


export async function Sim(width, height, ctx) {
    const gpu = new GPU()
    await gpu.init(width,height,ctx)
    
    let bitmapIds = {'-1':-1}

    let [bitmapData, meshData] = await db.transact(['bitmaps','meshes'],'readonly',
        async tx => await Promise.all([tx.query('bitmaps'), tx.query('meshes')]))

    let bitmaps = []
    for (let [bidx,[bid,bdata]] of enumerate(bitmapData)) {
        bitmapIds[bid] = bidx
        bitmaps.push(bdata.data)
    }

    meshData = [...meshData]
    if (meshData.length == 0)
	meshData = [[-1,{...MESH_DEFAULTS}]]
        
    let meshes = [], particles = [], pverts = [], tris = [], clusters = [], w0 = []
    for (let [midx,[mid,mdata]] of enumerate(meshData)) {
        let mesh = Mesh.alloc()
        mesh.color = v4(...mdata.color)
        mesh.pcolor = particleColors[midx % particleColors.length]
        mesh.volstiff = mdata.volstiff
        mesh.shearstiff = mdata.shearstiff
        mesh.tex = bitmapIds[mdata.bitmapId]
        mesh.pi = particles.length
        mesh.fluid = mdata['fluid']
        let { scale, offset } = mdata
        let g = await db.transact(db.storeNames,'readwrite',async tx => await tx.meshGeometry(mid,scale||1,offset||0))
        if (g.verts.length == 0)
            g.verts = [new GeoVert(0, v3(0,0,1))]

        //let vol = g.volume()
        //let area = g.surfarea()
        //let thickness = vol / area

        for (let vert of g.verts) {
            let p = Particle.alloc()
            p.x = p.x0 = vert.x
            p.w = mdata.fixed ? 0 : 1;
            p.friction = mdata.friction
            p.collision = mdata.collision
            particles.push(p)
            w0.push(p.w)

            let pv = ParticleVert.alloc()
            pv.mesh = midx
            for (let edge of vert.edges)
                pv.edges[pv.nedges++] = mesh.pi + edge.vert.id
            pverts.push(pv)
        }

        let mclusts = g.cluster()
        for (let verts of mclusts) {
            let cluster = Cluster.alloc()
            cluster.mesh = midx
            let c = [0,0,0]
            for (let vert of verts) {
                cluster.pids[cluster.n++] = vert.id + mesh.pi
                c = [c[0] + vert.x.x, c[1] + vert.x.y, c[2] + vert.x.z]
            }                            
            c = c.map(val => val / verts.length)
            cluster.c0 = v3(...c)
            let Q = M3js.of([0,0,0],[0,0,0],[0,0,0])
            for (let vert of verts) {
                let rx = vert.x.x - c[0], ry = vert.x.y - c[1], rz = vert.x.z - c[2]
                Q = Q.add([[rx*rx, rx*ry, rx*rz], [ry*rx, ry*ry, ry*rz], [rz*rx, rz*ry, rz*rz]])                
            }
            let qsum = Q[0].sum() + Q[1].sum() + Q[2].sum()
            if (qsum > 0) {
                let s = 1/qsum
                let Qs = Q.mulc(s)
                cluster.qinv = M3.of(Qs.invert())
                cluster.s = s
            }           
            clusters.push(cluster)
        }

        mesh.pf = particles.length
        for (let tri of g.tris) {
            let tvs = tri.verts.map((v,i) => TriVert.of(v3(0), v.id + mesh.pi, v3(0), midx, v2(...tri.uvs[i])))
            tris.push(Triangle.of(...tvs))
        }
        meshes.push(mesh)
    }

    meshes = Meshes.of(meshes)
    particles = Particles.of(particles)
    pverts = ParticleVerts.of(pverts)
    tris = Triangles.of(tris)
    clusters = Clusters.of(clusters)
    w0 = GPU.array({ type:f32, length:w0.length }).of(w0)

   
    let diameter = -1
    for (let p of particles)
        for (let i of range(p.nedges))
            diameter = max(diameter, p.x.dist(particles[p.edges[i]].x))
    if (diameter < 0)
        diameter = 0.02

    const uni = Uniforms.alloc()
    uni.seed = 666
    uni.selection = -1
    uni.grab_pid = -1

    let eye = gpu.buf({ type:Eye, usage:'UNIFORM|COPY_DST' })
    const Lights = GPU.array({ type:Light, length:6 })
    const lights = Lights.alloc()
    let lightEyes = range(lights.length).map(i => gpu.buf({ type:Eye, usage:'UNIFORM|COPY_DST' }))
    let colorTex, depthTex
    const setAttachments = () => {
        if (colorTex) colorTex.destroy()
        if (depthTex) depthTex.destroy()
        let size=[width,height], sampleCount=render.samples
        let usage=GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING
        colorTex = gpu.texture({ label:'colorAttach', size, sampleCount, format:render.format, usage })
        depthTex = gpu.texture({ label:'depthAttach', size, sampleCount, format:render.depth_fmt, usage })            
    }

    let cellLength = CELLSIZE * CELLDIM**3
    let cells = new Int32Array(cellLength)
    for (let i = 0; i < cellLength; i++) cells[i] = -1
    cells = new Cells(cells.buffer)
    
    const threads = gpu.threads
    let grab_hits = GrabHits.alloc()


    
    const bufs = Object.fromEntries([
        gpu.buf({ label:'pbuf', data:particles, usage:'STORAGE|COPY_SRC|COPY_DST|VERTEX' }),
        gpu.buf({ label:'vbuf', data:pverts, usage:'STORAGE|COPY_SRC|COPY_DST|VERTEX' }),
        gpu.buf({ label:'mbuf', data:meshes, usage:'STORAGE|COPY_DST|COPY_SRC' }),
        gpu.buf({ label:'cbuf', data:clusters, usage:'STORAGE|COPY_DST|COPY_SRC' }),
        gpu.buf({ label:'u', data:uni, usage:'UNIFORM|COPY_DST' }),
        gpu.buf({ label:'tribuf', data:tris, usage:'STORAGE|VERTEX|COPY_DST' }),
        gpu.buf({ label:'lbuf', data:lights, usage:'UNIFORM|FRAGMENT|COPY_DST' }),
        gpu.buf({ label:'grab_hits', data:grab_hits, usage:'STORAGE|COPY_SRC|COPY_DST' }),
        gpu.buf({ label:'w0', data:w0, usage:'STORAGE|COPY_SRC|COPY_DST' }),
        gpu.buf({ label:'cells', data:cells, usage:'STORAGE|COPY_DST|COPY_SRC' }),
    ].map(buf => [buf.label, buf]))


    
    const pd = ceil(particles.length/gpu.threads), pd2 = ceil(pd / gpu.threads)
   
    const syncUniforms = () => {
        uni.r = phys.r * diameter/2
        uni.cam_x = render.cam_x
        uni.spacemin = v3(phys.xmin, phys.ymin, phys.zmin)
        uni.spacemax = v3(phys.xmax, phys.ymax, phys.zmax)
        uni.damp = phys.damp
        let tscale = 60*phys.frameratio
        uni.a = v3(0,0,phys.a/(tscale**2));
        uni.friction = phys.friction
        uni.collision = phys.collision
        uni.volstiff = phys.volstiff
        uni.shearstiff = phys.shearstiff       
        return gpu.write(bufs.u, uni)
    }
    
    async function Computer() {
        
        let fwdstep = false
        let tstart = clock(), tlast = clock()
        let frames = 0, steps = 0

        const threads = gpu.threads
        const wgsl = (await fetchtext('./compute.wgsl')).interp({ threads, CELLSIZE, CELLDIM })

        const shader = await gpu.shader({
            compute:true, wgsl:wgsl, defs:[Particle, Mesh, Uniforms, GrabHit, GrabHits, Cluster],
            storage:{
                pbuf:Particles, mbuf:Meshes, grab_hits:GrabHits, setdps:v3uarr, cells:Cells,
                cbuf:Clusters, pavg:v3arr, vavg:v3arr, lavg:v3arr, iavg:m3arr, w0:f32arr,
            },
            uniform:{ u:Uniforms }
        })
    
        const { u, mbuf, pbuf, cbuf, cells } = bufs

        const pipe = (...args) => gpu.computePipe(...args)
        let keys = obj => [...Object.keys(obj)]
        
        const sched = gpu.cmdScheduler()

        let pass = sched.computePass()
        
        let binds = { pbuf, u, cells, grab_hits:bufs.grab_hits, w0:bufs.w0 }
        pass.call({ pipe:pipe({ shader, entryPoint:'predict', binds:keys(binds) }), dispatch:pd, binds })
        pass.stamp('predict')                              
                
        binds = { pbuf, u, cells }
        pass.call({ pipe:pipe({ shader, entryPoint:'wall_collide', binds:keys(binds) }), dispatch:pd, binds })
        pass.stamp('wallcollide')

        binds = { pbuf, u, cells }
        pass.call({ pipe:pipe({ shader, entryPoint:'pair_collide', binds:keys(binds) }), dispatch:pd, binds })
        pass.stamp('paircollide')

        

        binds = { pbuf, u }
        pass.call({ pipe:pipe({ shader, entryPoint:'xvupd', binds:keys(binds) }), dispatch:pd, binds })
        pass.stamp('xvupd')
                                                          
        
        binds = { pbuf, mbuf, u, cbuf }
        pass.call({ pipe:pipe({ shader, entryPoint:'surfmatch', binds:keys(binds) }), dispatch:1, binds })
        pass.stamp('surface match')
        
        let mbufs = []
        for (let [i,m] of enumerate(meshes)) {
            let n = m.pf - m.pi
            if (n <= 0) continue
            mbufs.push({
                pavg: gpu.buf({ label:'pavg', type:v3arr, size:v3arr.stride*n, usage:'STORAGE' }),
                vavg: gpu.buf({ label:'vavg', type:v3arr, size:v3arr.stride*n, usage:'STORAGE' }),
                lavg: gpu.buf({ label:'lavg', type:v3arr, size:m3arr.stride*n, usage:'STORAGE' }),
                iavg: gpu.buf({ label:'iavg', type:m3arr, size:m3arr.stride*n, usage:'STORAGE' }),
                mbuf: gpu.offset(bufs.mbuf, Meshes.stride * i)            
            })
        }
        let avgs_prep = pipe({ shader, entryPoint:'avgs_prep', binds:['mbuf','pbuf','pavg','vavg','lavg','iavg']})
        let avgs_calc = pipe({ shader, entryPoint:'avgs_calc', binds:['mbuf','pavg','vavg','lavg','iavg']})
        
        const meshavgs = (i) => {
            let n = meshes[i].pf - meshes[i].pi
            if (n <= 0) return []
            let pdm1 = ceil(n / threads), pdm2 = ceil(pdm1 / threads)
            pass.call({ pipe:avgs_prep, dispatch:pdm1, binds:{ pbuf, ...mbufs[i] }})
            pass.call({ pipe:avgs_calc, dispatch:pdm1, binds:mbufs[i] })
            if (pdm1 > 1) {
                pass.call({ pipe:avgs_calc, dispatch:pdm2, binds:mbufs[i] })
                if (pdm2 > 1) pass.call({ pipe:avgs_calc, dispatch:1, binds:mbufs[i]})
            }
        }

        for (const i of range(meshes.length)) meshavgs(i)
        for (const i of range(meshes.length)) meshavgs(i)
        pass.stamp('mesh avgs')

        return {
            stats: async () => {
                let ret = { kind:'phys', fps:frames / (tlast - tstart) }
                if (!gpu.alive) return ret
                ret.profile = await sched.queryStamps()
                tstart = tlast
                frames = 0
                return ret
            },

            step: async () => {
                if (particles.length == 0) return
                if (!phys.paused || fwdstep) {
                    uni.t += 1
                    uni.seed = uni.seed * 16807 % 2147483647
                    syncUniforms()
                    sched.execute()
                    checkGrabHits()
                    frames++
                    steps++
                }
                tlast = clock()
                fwdstep = false
            },

            fwdstep: () => {
                fwdstep = true
            }
        }

    }
    
    async function Renderer() {

        let reset = false
        let tstart = clock(), tlast = clock()
        let frames = 0
        let sched
        
        const setup = async () => {
            gpu.configure(ctx, width, height, render)
            render.watch(render.keys.filter(key => !['fov','cam_x','cam_ud','cam_lr','dir','lfov','near','far'].includes(key)), () => {
                reset = true
            })

            const td = ceil(tris.length/gpu.threads)

            let wgsl = (await fetchtext('./prerender.wgsl')).interp({threads: gpu.threads})
            const preShader = await gpu.shader({
                wgsl, compute: true, defs:[ Mesh, Particle, Uniforms, TriVert, Triangle, ParticleVert ],
                storage:{ pbuf:Particles, vbuf:ParticleVerts, mbuf:Meshes, tribuf:Triangles },
                uniform: { u:Uniforms }
            })

            wgsl = await fetchtext('./render.wgsl')
            wgsl = wgsl.interp({numLights: lights.length })
            const shader = await gpu.shader({
                wgsl, defs:[Mesh, Uniforms, Light, Eye],
                storage:{ mbuf:Meshes },
                uniform:{ u:Uniforms, lbuf:Lights, eye:Eye },
                textures:{
                    tex:{ name:'texture_2d_array<f32>' },
                    shadowMaps:{ name:'texture_depth_2d_array', sampleType:'depth' }
                },
                samplers:{ texSamp:{ name:'sampler' }, shadowSamp:{ name:'sampler_comparison' } }
            })
            
            const { u, mbuf, pbuf, vbuf, lbuf, tribuf } = bufs
            let keys = obj => [...Object.keys(obj)]
            let binds,dispatch,pipe,desc
            
            sched = gpu.cmdScheduler()

            const precomp = sched.computePass()
                       
            if (particles.length > 0 && td > 0) {
                binds = { pbuf, vbuf, u }
                const normals = gpu.computePipe({ shader:preShader, entryPoint:'normals', binds:keys(binds) })
                precomp.call({ pipe:normals, dispatch:pd, binds })
                precomp.stamp('normals')
            }

            if (td > 0) {
                binds = { pbuf, vbuf, tribuf }
                const updTris = gpu.computePipe({ shader:preShader, entryPoint:'update_tris', binds:keys(binds) })
                precomp.call({ pipe:updTris, dispatch:td, binds })
                precomp.stamp('updatetris')
            }

            let tex = gpu.bitmapTexture(bitmaps)
            let texSamp = gpu.sampler({ magFilter: 'linear', minFilter: 'linear'})
            let shadowSamp = gpu.sampler({ compare:'less', magFilter: 'linear', minFilter: 'linear' })
            let usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
            let shadowDepth = gpu.texture({ size: [SHADOWRES, SHADOWRES, max(2,lights.length)], format:render.depth_fmt, usage })
            let shadowMaps = { resource: shadowDepth.createView() }

            setAttachments()

            let shadowPasses
            if (render.shadows) {
                shadowPasses = range(lights.length).map(i => sched.drawPass(() => ({
                    colorAttachments:[],
                    depthStencilAttachment: {
                        view:shadowDepth.createView({ baseArrayLayer:i, arrayLayerCount:1 }),
                        depthClearValue: 1, depthLoadOp:'clear', depthStoreOp:'store'
                    }
                })))            
                sched.stamp('shadowpasses')
            }

            let ms = render.samples > 1
            let renderPass = sched.drawPass(() => {
                let ctxView = ctx.getCurrentTexture().createView()
                return {
                    colorAttachments: [{
                        view: ms ? colorTex.createView() : ctxView,
                        resolveTarget: ms ? ctxView : undefined,
                        loadOp:'clear', storeOp:'store',
                        clearValue:{ r:0,g:0,b:0,a:0 }
                    }],
                    depthStencilAttachment: {
                        view:depthTex.createView({label:'depthAttach'}),
                        depthClearValue:1, depthLoadOp:'clear', depthStoreOp:'store',
                    }
                }
            })
            
            if (tris.length > 0 && render.surfaces) {
                dispatch = tris.length * 3
                desc = {
                    shader, entry:'surface',
                    vertBufs: [{ buf:tribuf, arrayStride:TriVert.size,
                                 attributes: [{ shaderLocation:0, offset:TriVert.x.off, format:'float32x3' },
                                              { shaderLocation:1, offset:TriVert.norm.off, format:'float32x3' },
                                              { shaderLocation:2, offset:TriVert.mesh.off, format:'uint32' },
                                              { shaderLocation:3, offset:TriVert.uv.off, format:'float32x2' }]}]
                }
                
                binds = { mbuf, u, eye, lbuf, tex, texSamp }
                if (render.shadows) {
                    pipe = gpu.renderPipe({ ...desc, binds:keys(binds), frag:false, samples:1 })
                    for (let i of range(lights.length))
                        shadowPasses[i].draw({ pipe, dispatch, binds:{ ...binds, eye:lightEyes[i] }})
                }
                binds = { ...binds, shadowMaps, shadowSamp }
                pipe = gpu.renderPipe({ ...desc, binds:keys(binds) })
                renderPass.draw({ pipe, dispatch, binds })
                renderPass.stamp('triangles')
            }

            if (render.particles && particles.length > 0) {
                dispatch = [3, particles.length]                              
                desc = {
                    shader, entry:'particle', topology:'triangle-list',
                    vertBufs: [{ buf:pbuf, arrayStride:Particles.stride, stepMode: 'instance',
                                 attributes: [{ shaderLocation:0, offset:Particle.x.off, format:'float32x3' }]},
                               { buf:vbuf, arrayStride:ParticleVerts.stride, stepMode:'instance',
                                 attributes: [{ shaderLocation:1, offset:ParticleVert.mesh.off, format:'uint32' }]}]
                }
                binds = { mbuf, u, eye, lbuf }
                if (render.shadows) {
                    pipe = gpu.renderPipe({ ...desc, binds:keys(binds), frag:'depth', samples:1 })
                    for (let i of range(lights.length))
                        shadowPasses[i].draw({ pipe, dispatch, binds:{ ...binds, eye:lightEyes[i] }})
                }
                binds = { ...binds, shadowMaps, shadowSamp }
                pipe = gpu.renderPipe({ ...desc, binds:keys(binds) })
                renderPass.draw({ pipe, dispatch, binds })
                renderPass.stamp('particles')            
            }

            

            if (render.walls) {
                dispatch = 36
                desc = { shader, entry:'walls', cullMode:'front', topology:'triangle-list' }               
                binds = { u, eye, lbuf }
                if (render.shadows) {
                    pipe = gpu.renderPipe({ ...desc, binds:keys(binds), frag:false, samples:1 })
                    for (let i of range(lights.length))
                        shadowPasses[i].draw({ pipe, dispatch, binds:{ ...binds, eye:lightEyes[i] }})
                }
                binds = { ...binds, shadowMaps, shadowSamp }
                pipe = gpu.renderPipe({ ...desc, binds:keys(binds) })
                renderPass.draw({ pipe, dispatch, binds })
                renderPass.stamp('walls')
            }

            if (render.normals && tris.length > 0) {
                binds = { u, eye }
                pipe = gpu.renderPipe({ shader, entry:'vnormals', binds:keys(binds), topology: 'line-list',
                                        vertBufs: [{ buf:tribuf, arrayStride:TriVert.size, stepMode: 'instance',
                                                     attributes: [{ shaderLocation:0, offset:TriVert.x.off, format:'float32x3' },
                                                                  { shaderLocation:1, offset:TriVert.norm.off, format:'float32x3' }]}]})
                renderPass.draw({ pipe, dispatch:[2, tris.length*3], binds })
                renderPass.stamp('normals')
            }

            if (render.velocity && render.particles && particles.length > 0) {
                binds = { u, eye }                
                pipe = gpu.renderPipe({ shader, entry:'velocity', binds:keys(binds), topology: 'line-list', depthCompare:'always',
                                        vertBufs: [{ buf:pbuf, arrayStride:Particles.stride, stepMode: 'instance',
                                                     attributes: [{ shaderLocation:0, offset:Particle.x.off, format:'float32x3' },
                                                                  { shaderLocation:1, offset:Particle.v.off, format:'float32x3' }]}]})
                renderPass.draw({ pipe, dispatch:[2, particles.length], binds })
                renderPass.stamp('normals')
            }

            if (render.lights) {
                let binds = { u, eye, lbuf }
                let pipe = gpu.renderPipe({ shader, entry:'lights', binds:keys(binds), topology: 'triangle-list',
                                            atc:false, depthWriteEnabled:false })
                renderPass.draw({ pipe, dispatch:[3, lights.length], binds })
                renderPass.stamp('lights')
            }           

        }

        await setup()

        const syncEyes = () => {
            let bmin = v3(phys.xmin, phys.ymin, phys.zmin)
            let bmax = v3(phys.xmax, phys.ymax, phys.zmax)
            let x = bmin.add(bmax).divc(2)
            let b = bmax.sub(bmin).divc(2)
            let i = 0;
            for (const [fwd,v,h] of [[0,2,1],[1,2,0],[2,1,0]]) {
                for (const s of [-1,1]) {
                    const l = lights[i++]
                    l.x = x
                    l.color = v3(1.0,0.98,0.95)
                    l.dir[fwd] = s
                    l.power = 1
                    l.viewproj = M4.box(b[h]/b[fwd], b[v]/b[fwd], 0.0001, b[fwd]).mul(M4.look(l.x, l.dir))
                    l.shadow = render.shadows ? 1 : 0;
                }
            }            
            gpu.write(bufs.lbuf, lights)
            gpu.write(eye, Eye.of(eyeProj().mul(eyeView()), render.cam_x, eyeDir()))
            for (let [i,l] of enumerate(lights))
                gpu.write(lightEyes[i], Eye.of(l.viewproj, l.x, l.dir))
        }

        return {
            stats: async () => {               
                const ret = { kind:'render', fps: frames / (tlast - tstart) }
                if (!gpu.alive) return ret;
                if (sched)
                    ret.profile = await sched.queryStamps()                
                tstart = tlast
                frames = 0
                return ret
            },

            step: async () => {
                if (reset) {
                    await gpu.dev.queue.onSubmittedWorkDone()
                    await setup()
                    reset = false
                }

                syncUniforms()
                syncEyes()
                gpu.write(bufs.lbuf, lights)
                sched.execute()
                frames++
                tlast = clock()
            }
        }
    }
    
    const dbgbuf = async (buf) => {
        let type = (typeof buf.type == 'function') ? buf.type : Int32Array
        let data = new type(await gpu.read(buf))
        window['d'+buf.label] = data
        return data
    }

    const eyeDir = () => v3(sin(render.cam_lr) * cos(render.cam_ud),
                            cos(render.cam_lr) * cos(render.cam_ud),
                            sin(render.cam_ud)).normalized()    
    const eyeProj = () => {
        return M4.perspective(render.fov, width/height, 0.01, 100)
    }
    const eyeView = () => M4.look(render.cam_x, eyeDir())

    const clipToRay = (x,y) => {
        let clip = v4(2*x/width - 1, 1 - 2*y/height,-1,1)
        let eye = eyeProj().inverse().transform(clip)
        let ray = eyeView().inverse().transform(v4(eye.x,eye.y,-1,0))
        return v3(ray.x,ray.y,ray.z).normalized()
    }

    function resize(w, h) {
        width = w
        height = h
        gpu.configure(ctx,w,h,render)        
        setAttachments()
    }

    async function run() {
        while (gpu.alive) {
            const p = new Promise(resolve => requestAnimationFrame(resolve))
            const tstart = clock()
            await renderer.step()
            for (let i of range(phys.frameratio))
                await computer.step()
            await p
        }        
    }

    function setCam(x, lr, ud) {
        render.cam_x = x
        render.cam_lr = lr
        render.cam_ud = ud
    }
    
    function rotateCam(dx, dy) {
        render.cam_lr += dx
        render.cam_ud = clamp(render.cam_ud + dy, -PI / 2.1, PI / 2.1)
    }

    function strafeCam(dx, dy) {
        const delta = v3(-dx * cos(render.cam_lr), dx * sin(render.cam_lr), -dy)
        render.cam_x = render.cam_x.add(delta)
    }

    function advanceCam(delta) {
        const camDir = v3(sin(render.cam_lr) * cos(render.cam_ud),
                          cos(render.cam_lr) * cos(render.cam_ud),
                          sin(render.cam_ud)).normalized()
        render.cam_x = render.cam_x.add(v3(delta * camDir.x, delta * camDir.y, delta * camDir.z))
    }
    
    let grabDepth, grabOffset, grabCallback

    function grabParticle(x, y, cb) {
        uni.grab_ray = clipToRay(x,y)
        uni.grab_query = 1
        uni.grab_pid = -1
        return new Promise(res => { grabCallback = res })
    }

    async function checkGrabHits() {
        if (uni.grab_query == 0) return
        uni.grab_query = 0
        uni.selection = -1
        grab_hits = new GrabHits(await gpu.read(bufs.grab_hits))
        if (grab_hits.len == 0) return
        let hit = grab_hits.list[range(grab_hits.len).min(i => grab_hits.list[i].x.sub(render.cam_x).mag())]
        uni.selection = uni.grab_pid = hit.pid
        uni.grab_x = hit.x.clamp(uni.spacemin.addc(uni.r), uni.spacemax)
        let fwd = v3(sin(render.cam_lr), cos(render.cam_lr), 0).normalized()
        grabDepth = hit.x.sub(render.cam_x).dot(fwd)
        grabOffset = uni.grab_x.sub(render.cam_x.add(uni.grab_ray.mulc(grabDepth/uni.grab_ray.dot(fwd))))
        grab_hits.len = 0
        gpu.write(bufs.grab_hits, grab_hits)
        grabCallback()
        dbg({ pid:uni.selection, x:hit.x })
    }
        
    function moveParticle(x, y) {
        if (uni.grab_pid == -1) return
        let r = clipToRay(x,y)
        let fwd = v3(sin(render.cam_lr), cos(render.cam_lr), 0).normalized()
        uni.grab_x = render.cam_x.add(r.mulc(grabDepth/r.dot(fwd))).add(grabOffset)
        uni.grab_x = uni.grab_x.max(uni.spacemin.addc(uni.r))
        uni.grab_x = uni.grab_x.min(uni.spacemax.subc(uni.r))
    }

    function dropParticle() {
        uni.grab_pid = -1
    }

    async function fixParticle() {
        let pid = uni.selection
        if (pid < 0) return
        dbg({fixing:pid})
        w0[pid] = w0[pid] == 0 ? 1 : 0
        gpu.write(bufs.w0, w0)
    }


    console.time('compute init')
    const computer = await Computer()
    console.timeEnd('compute init')
    console.time('renderer init')
    const renderer = await Renderer()
    console.timeEnd('renderer init')

    dbg({ nparts:particles.length, nmeshes:meshes.length, ntris:tris.length, nclusters:clusters.length, r:diameter/2 })
    return scope(stmt => eval(stmt))
}


export const loadWavefront = async (name, data, transaction) => {
    let meshes = transaction.objectStore('meshes')
    let meshId = await transaction.wait(meshes.add({ ...MESH_DEFAULTS, name }))
    let verts = transaction.objectStore('verts')
    let faces = transaction.objectStore('faces')
    let localVerts = 1
    let vertIds = {}
    let localUVs = 1
    let uvIds = {}
    for (let line of data.split(/[\r\n]/))  {
        let [key, ...toks] = line.split(/\s/)
        if (key == 'v') {
            let vdata = toks.map(parseFloat)
            let x = vdata.slice(0, 3)
            vertIds[localVerts++] = await transaction.wait(verts.add({ x, meshId }))
        } else if (key == 'vt') {
            let vtdata = toks.map(parseFloat)
            uvIds[localUVs++] = vtdata.slice(0, 2)
        } else if (key == 'f') {
            let face = toks.map((tok,i) => [`vertId${i}`, vertIds[parseInt(tok.split('/')[0])]])
            let uv = toks.map(tok => uvIds[parseInt(tok.split('/')[1])] || [0,0])
            await transaction.wait(faces.add({ ...Object.fromEntries(face), uv, meshId }))
        }
    }
}

export const loadBitmap = async (name, bitmap, transaction) => {
    let bitmaps = transaction.objectStore('bitmaps')
    await transaction.wait(bitmaps.add({ name, data: bitmap }))
}






