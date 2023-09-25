import * as util from './utils.js'
import * as gpu from './gpu.js'
import * as geo from './geometry.js'
Object.assign(globalThis, util, gpu, geo)

const MAXEDGES = 18
const MAXRING = 48
const CELLSIZE = 8
const CELLDIM = 256

const particleColors = [v4(.3,.6,.8,1), v4(.99,.44,.57, 1), v4(.9, .48, .48, 1)]
const LIGHTS = [
    { power:1, color:v3(1,1,1), x:v3(2,2,2.3) },
    { power:1, color:v3(1,1,1), x:v3(2,-2,2.3) },
    { power:1, color:v3(1,1,1), x:v3(-2,2,2.3) },
    { power:1, color:v3(1,1,1), x:v3(-2,-2,2.3) },
]

const MESH_DEFAULTS = {
    name:'default', bitmapId:-1, color:[1,1,1,1], offset:[0,0,0], rotation:[0,0,0],
    scale:[1,1,1], volstiff:1, shearstiff:1, friction:1, collision:1, fluid:0, w:1,
}

export const phys = new Preferences('phys')
phys.addBool('paused', false)
phys.addNum('r', 1.0, 0.1, 10, 0.1)
phys.addNum('frameratio', 3, 1, 20, 1)
phys.addNum('speed', 1, 0.05, 5, .01)
phys.addNum('gravity', 9.8, -5, 20, 0.1)
phys.addNum('volstiff', .5, 0, 2, 0.001)
phys.addNum('shearstiff', .5, 0, 1, 0.001)
phys.addNum('damp', 0.5, -100, 100, .1)
phys.addNum('friction', 1, 0, 1, .01)
phys.addNum('collision', 1, 0, 1, .01)
phys.addNum('xmin', -25, -25, 0, 0.1)
phys.addNum('xmax', 25, 0.1, 25, 0.1)
phys.addNum('ymin', -25, -25, 0, 0.1)
phys.addNum('ymax', 25, .1, 25, 0.1)
phys.addNum('zmin', 0, -25, 0, 0.1)
phys.addNum('zmax', 50, 0.1, 50, 0.1)



export const render = new Preferences('render')
render.addBool('particles', true)
render.addBool('walls', true)
render.addBool('normals', false)
render.addBool('depth_wr', true)
render.addBool('atc', true)
render.addChoice('depth_cmp', 'less', ['less-equal','less','greater-equal','greater','always','never'])
render.addChoice('cull', 'back', ['none','back','front'])
render.addChoice('alpha_mode', 'premultiplied', ['opaque','premultiplied'])
render.addChoice('color_op', 'add', ['add','subtract','reverse-subtract','min','max'])
render.addChoice('alpha_op', 'add', ['add','subtract','reverse-subtract','min','max'])
const factors = ['zero','one','src','one-minus-src','src-alpha','one-minus-src-alpha',
                 'dst','one-minus-dst','dst-alpha','one-minus-dst-alpha']
render.addChoice('color_src', 'one', factors)
render.addChoice('color_dst', 'one-minus-src-alpha', factors)
render.addChoice('alpha_src', 'one', factors)
render.addChoice('alpha_dst', 'one-minus-src-alpha', factors)
render.addChoice('format', 'rgba16float', ['rgba8unorm','bgra8unorm','rgba16float'])
render.addChoice('depth_fmt', 'depth32float-stencil8',
                 ['depth16unorm','depth24plus','depth24plus-stencil8','depth32float','depth32float-stencil8'])
render.addNum('samples', 4, 1, 4, 3)
render.addNum('fov', 60, 1, 150, 1)
render.addVector('cam_x', v3(0, -2, 2))
render.addNum('cam_ud', 0)
render.addNum('cam_lr', 0)
render.hidden['cam_x'] = true
render.hidden['cam_ud'] = true
render.hidden['cam_lr'] = true

const clock = () => phys.speed*performance.now()/1000

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
        ['xprev', V3],
        ['friction',f32],
        ['x0', V3],
        ['collision', f32],
        ['xupd', V3],
        ['hash',i32],
        ['v', V3],        
        ['vprev',V3],
        ['vupd',V3],
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
    



export const Light = GPU.struct({
    name:'Light',
    fields:[
        ['x', V3],
        ['dir', V3],
        ['power', f32],
        ['color', V3]
    ]
})

export const Uniforms = GPU.struct({
    name:'Uniforms',
    fields:[       
        ['proj', M4],
        ['view', M4],
        ['mvp', M4],
        ['cam_x', V3],
        ['cam_fwd', V3],
        ['width', i32],
        ['height', i32],
        ['dt', f32],
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

/*export const Bounds = GPU.struct({
    name:'Bounds',
    fields:[
        ['min', V3],
        ['max', V3],        
        ['grid', V3I],
        ['stride', V3I],
    ]
})*/

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

/*GPU.struct({
    name:'Cell',
    fields:[
        ['pids', GPU.array({ type:i32, length: 8 })],
    ]
})*/

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
    console.time('gpu init')
    const gpu = new GPU()
    await gpu.init(width,height,ctx)
    console.timeEnd('gpu init')
    
    let bitmapIds = {'-1':-1}

    console.time('db load')
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
            p.x = p.xprev = p.x0 = vert.x
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

    const lights = GPU.array({ type:Light, length:LIGHTS.length }).alloc(LIGHTS.length)
    for (const [i,l] of enumerate(LIGHTS)) {
        lights[i].color = l.color
        lights[i].x = l.x
        lights[i].power = l.power
    }

    let cells = new Int32Array(CELLSIZE * CELLDIM**3)
    for (let i of range(cells.length)) cells[i] = -1
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
        gpu.buf({ label:'cells', data:cells, usage:'STORAGE|COPY_DST|COPY_SRC' })        
    ].map(buf => [buf.label, buf]))

    const pd = ceil(particles.length/gpu.threads), pd2 = ceil(pd / gpu.threads)

    console.timeEnd('db load')
    
    const syncUniforms = () => {
        uni.r = phys.r * diameter/2
        uni.cam_x = render.cam_x
        uni.width = width
        uni.height = height
        uni.cam_fwd = v3(sin(render.cam_lr) * cos(render.cam_ud),
                              cos(render.cam_lr) * cos(render.cam_ud),
                              sin(render.cam_ud)).normalized()
        uni.proj = M4.perspective(render.fov, width/height, .001, 100)
        uni.view = M4.look(render.cam_x, uni.cam_fwd, v3(0,0,1))
        uni.mvp = uni.view.mul(uni.proj)
        uni.spacemin = v3(phys.xmin, phys.ymin, phys.zmin)
        uni.spacemax = v3(phys.xmax, phys.ymax, phys.zmax)
        uni.damp = phys.damp
        uni.a = v3(0,0,-phys.gravity);
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

        const cmds = []
        const pass = (...args) => { cmds.push(gpu.computePass(...args)) }
        const pipe = (...args) => gpu.computePipe(...args)
        const stamp = (tag) => { cmds.push(gpu.timestamp(tag)) }
        let keys = obj => [...Object.keys(obj)]       
        

        stamp('')

        
        let binds = { pbuf, u, cells, grab_hits:bufs.grab_hits, w0:bufs.w0 }
        pass({ pipe:pipe({ shader, entryPoint:'predict', binds:keys(binds) }), dispatch:pd, binds })
        stamp('predict')                              
                
        binds = { pbuf, u, cells }
        pass({ pipe:pipe({ shader, entryPoint:'collide', binds:keys(binds) }), dispatch:pd, binds })
        stamp('collide')

        binds = { pbuf, u }
        pass({ pipe:pipe({ shader, entryPoint:'xvupd', binds:keys(binds) }), dispatch:pd, binds })
        stamp('xvupd')
                       
                                   
        binds = { pbuf, mbuf, u, cbuf }
        pass({ pipe:pipe({ shader, entryPoint:'surfmatch', binds:keys(binds) }), dispatch:1, binds })
        stamp('surface match')

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
            pass({ pipe:avgs_prep, dispatch:pdm1, binds:{ pbuf, ...mbufs[i] }})
            pass({ pipe:avgs_calc, dispatch:pdm1, binds:mbufs[i] })
            if (pdm1 > 1) {
                pass({ pipe:avgs_calc, dispatch:pdm2, binds:mbufs[i] })
                if (pdm2 > 1) pass({ pipe:avgs_calc, dispatch:1, binds:mbufs[i]})
            }
        }

        for (const i of range(meshes.length)) meshavgs(i)
        for (const i of range(meshes.length)) meshavgs(i)
        stamp('mesh avgs')        
        
        const batch = gpu.encode(cmds)

        let hs = [0,0]
        
        return {
            stats: async () => {
                let ret = { kind:'phys', fps:frames / (tlast - tstart) * phys.speed }
                if (!gpu.alive) return ret                
                if (batch) {
                    let data = new BigInt64Array(await gpu.read(batch.stampBuf))
                    let labels = batch.stampLabels
                    ret.profile = [...range(1,labels.length)].map(i => [labels[i], data[i] - data[i-1]])
                }
                tstart = tlast
                frames = 0
                return ret
            },

            step: async tstep => {
                if (particles.length == 0) return
                if (!phys.paused || fwdstep) {
                    uni.dt = tstep
                    uni.t += tstep
                    uni.seed = uni.seed * 16807 % 2147483647
                    syncUniforms()
                    batch.execute()
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
        let batch

        const setup = async () => {
            gpu.configure(ctx, width, height, render)
            render.watch(render.keys.filter(key => !['fov','cam_x','cam_ud','cam_lr'].includes(key)), () => {
                if (render.color_op == 'max' || render.color_op == 'min')
                    render.color_src = render.color_dst = 'one'
                if (render.alpha_op == 'max' || render.alpha_op == 'min')
                    render.alpha_src = render.alpha_dst = 'one'
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
                wgsl, defs:[Mesh, Uniforms, Light],
                storage:{ mbuf:Meshes },
                uniform:{ u:Uniforms, lbuf:lights.constructor },
                textures:{ tex:{ name:'texture_2d_array<f32>' } },
                samplers:{ samp:{ name:'sampler' } }
            })
            
            let tex = gpu.texture(bitmaps)
            let samp = gpu.sampler()

            const { u, mbuf, pbuf, vbuf, lbuf, tribuf } = bufs
            let keys = obj => [...Object.keys(obj)]
            
            const cmds = []
            cmds.push(gpu.timestamp(''))

            
            if (particles.length > 0 && td > 0) {
                let binds = { pbuf, vbuf, u }
                const normals = gpu.computePipe({ shader:preShader, entryPoint:'normals', binds:keys(binds) })
                cmds.push(gpu.computePass({ pipe:normals, dispatch:pd, binds }))
                cmds.push(gpu.timestamp('normals'))
            }

            if (td > 0) {
                let binds = { pbuf, vbuf, tribuf }
                const updTris = gpu.computePipe({ shader:preShader, entryPoint:'update_tris', binds:keys(binds) })
                cmds.push(gpu.computePass({ pipe:updTris, dispatch:td, binds }))
                cmds.push(gpu.timestamp('updatetris'))
            }
                

            const draws = []
            if (render.walls) {
                let binds = { u, lbuf }
                let pipe = gpu.renderPipe({ shader, entry:'walls', cullMode:'back', binds:keys(binds), topology:'triangle-strip' })
                draws.push(gpu.draw({ pipe, dispatch:14, binds }))
            }

            if (tris.length > 0) {
                let binds = { mbuf, u, lbuf, tex, samp }
                let pipe = gpu.renderPipe({
                    shader, entry:'surface', binds:keys(binds),
                    vertBufs: [{ buf:tribuf, arrayStride:TriVert.size,
                                 attributes: [{ shaderLocation:0, offset:TriVert.x.off, format:'float32x3' },
                                              { shaderLocation:1, offset:TriVert.norm.off, format:'float32x3' },
                                              { shaderLocation:2, offset:TriVert.mesh.off, format:'uint32' },
                                              { shaderLocation:3, offset:TriVert.uv.off, format:'float32x2' }]}]
                })
                draws.push(gpu.draw({ pipe, dispatch:tris.length*3, binds }))
            }

            if (render.particles && particles.length > 0) {
                let binds = { mbuf, u, lbuf }
                let pipe = gpu.renderPipe({
                    shader, entry:'particle', binds:keys(binds), topology:'triangle-list',
                    vertBufs: [{ buf:pbuf, arrayStride:Particles.stride, stepMode: 'instance',
                                 attributes: [{ shaderLocation:0, offset:Particle.x.off, format:'float32x3' }]},
                               { buf:vbuf, arrayStride:ParticleVerts.stride, stepMode:'instance',
                                 attributes: [{ shaderLocation:1, offset:ParticleVert.mesh.off, format:'uint32' }]}]
                })
                draws.push(gpu.draw({ pipe, dispatch:[3, particles.length], binds }))
            }

            if (render.normals) {               
                let pipe = gpu.renderPipe({ shader, entry:'vnormals', binds:['u'], topology: 'line-list',
                    vertBufs: [{ buf:tribuf, arrayStride:TriVert.size, stepMode: 'instance',
                    attributes: [{ shaderLocation:0, offset:TriVert.x.off, format:'float32x3' },
                                 { shaderLocation:1, offset:TriVert.norm.off, format:'float32x3' }]}]})
                draws.push(gpu.draw({ pipe, dispatch:[2, tris.length*3], binds:{u} }))
            }

            let binds = { u, lbuf }
            let pipe = gpu.renderPipe({ shader, entry:'lights', binds:keys(binds), topology: 'triangle-list',
                                        atc:false, depthWriteEnabled:false })
            draws.push(gpu.draw({ pipe, dispatch:[3, lights.length], binds }))
            cmds.push(gpu.renderPass(draws))
            cmds.push(gpu.timestamp('draws'))
            
            batch = gpu.encode(cmds)
        }

        await setup()

        return {
            stats: async () => {               
                const ret = { kind:'render', fps: frames / (tlast - tstart) * phys.speed }
                if (!gpu.alive) return ret;
                if (batch) {
                    let data = new BigInt64Array(await gpu.read(batch.stampBuf))
                    let labels = batch.stampLabels
                    ret.profile = [...range(1,labels.length)].map(i => [labels[i], data[i] - data[i-1]])
                }
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
                gpu.write(bufs.lbuf, lights)
                batch.execute()
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

    const clipToRay = (x,y) => {
        let clip = v4(2*x/width - 1, 1 - 2*y/height,-1,1)
        let eye = uni.proj.inverse().transform(clip)
        let ray = uni.view.inverse().transform(v4(eye.x,eye.y,-1,0))
        return v3(ray.x,ray.y,ray.z).normalized()
    }

    function resize(w, h) {
        width = w
        height = h
        gpu.configure(ctx,w,h,render)

    }

    async function run() {
        while (gpu.alive) {
            const p = new Promise(resolve => requestAnimationFrame(resolve))
            const tstart = clock()
            await renderer.step()
            for (let i of range(phys.frameratio))
                await computer.step(1/60/phys.frameratio * phys.speed)
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
        render.cam_ud = clamp(render.cam_ud + dy, -PI / 2, PI / 2)
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






