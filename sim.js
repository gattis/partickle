import * as util from './utils.js'
import * as gpu from './gpu.js'
import * as geo from './geometry.js'
Object.assign(globalThis, util, gpu, geo)

const MAXNN = 32
const MAXEDGES = 18
const MAXRING = 64

const particleColors = [v4(.3,.6,.8,1), v4(.99,.44,.57, 1), v4(.9, .48, .48, 1)]
const LIGHTS = [
    { power:1.5, color:v3(1,.85,.6), pos:v3(2,2,2.3) },
    { power:1.5, color:v3(1,.85,.6), pos:v3(2,-2,2.3) },
    { power:1.5, color:v3(1,.85,.6), pos:v3(-2,2,2.3) },
    { power:1.5, color:v3(1,.85,.6), pos:v3(-2,-2,2.3) },
]

const MESH_DEFAULTS = {
    name:'default', bitmapId:-1, color:[1,1,1,1], offset:[0,0,0], rotation:[0,0,0], gravity:1, invmass:10,
    scale:[1,1,1], 'surf stiff':1, friction:1, 'collision damp':1, fluid:0, fixed: 0
}

export const phys = new Preferences('phys')
phys.addBool('paused', false)
phys.addNum('r', 1.0, 0.1, 10, 0.1)
phys.addNum('frameratio', 3, 1, 20, 1)
phys.addNum('speed', 1, 0.05, 5, .01)
phys.addNum('gravity', 9.8, -5, 20, 0.1)
phys.addNum('surf_stiff', .5, 0, 1, 0.001)
phys.addNum('friction', 0.1, 0, 1, .01)
phys.addNum('damp', 0.5, 0, 1, .01)
phys.addNum('collidamp', .1, 0, 1, .001)
phys.addNum('xspace', 100, 0, 100, 0.1)
phys.addNum('yspace', 100, 0, 100, 0.1)
phys.addNum('zspace', 100, 0, 100, 0.1)

export const render = new Preferences('render')
render.addBool('particles', true)
render.addBool('ground', true)
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
render.addVector('cam_pos', v3(0, -2, 2))
render.addNum('cam_ud', 0)
render.addNum('cam_lr', 0)
render.hidden['cam_pos'] = true
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
        ['gravity', f32],
        ['surf_stiff', f32],
        ['friction', f32],
        ['collidamp', f32],
        ['fluid', i32],
       // ['padding', GPU.array({ type:u32, length:13 })]
    ]
})

export const Particle = GPU.struct({
    name:'Particle',
    fields:[
        ['pos', V3],
        ['hash', i32],
        ['rest_pos', V3],
        ['mesh', u32],
        ['prev_pos', V3],
        ['fixed', u32],
        ['delta_pos', V3],
        ['k', u32],
        ['vel', V3],
        ['nedges', u32],
        ['norm', V3],
        ['nring',u32],        
        ['c0',V3],
        ['s',f32],
        ['qinv', M3],
        ['edges', GPU.array({ type:u32, length:MAXEDGES })],
        ['nn', GPU.array({ type:u32, length:MAXNN })],
        ['rings', GPU.array({ type:u32, length:MAXRING })]
    ]
})



export const Light = GPU.struct({
    name:'Light',
    fields:[
        ['pos', V3],
        ['dir', V3],
        ['power', f32],
        ['color', V3]
    ]
})

export const Uniforms = GPU.struct({
    name:'Uniforms',
    fields:[
        ...phys.keys.filter(k => phys.type[k] == 'num').map(k => [k, f32]),
        ['proj', M4],
        ['view', M4],
        ['mvp', M4],
        ['cam_pos', V3],
        ['cam_fwd', V3],
        ['width', i32],
        ['height', i32],
        ['dt', f32],
        ['t', f32],
        ['ground', u32],
        ['selection', i32],
        ['grabbing', i32],
        ['grabTarget', V3],
    ]
})

export const TriVert = GPU.struct({
    name:'TriVert',
    fields:[
        ['pos', V3],
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

const Meshes = GPU.array({ type:Mesh })
const Particles = GPU.array({ type:Particle })
const Triangles = GPU.array({ type:Triangle })

export async function Sim(width, height, ctx) {

    const gpu = new GPU()
    await gpu.init(width,height,ctx)

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
        
    let meshes = [], particles = [], tris = [], groups = [new Set()]
    for (let [midx,[mid,mdata]] of enumerate(meshData)) {
        let mesh = Mesh.alloc()
        mesh.color = v4(...mdata.color)
        mesh.pcolor = particleColors[midx % particleColors.length]
        mesh.gravity = mdata.gravity
        mesh.surf_stiff = mdata['surf stiff']
        mesh.friction = mdata.friction
        mesh.collidamp = mdata['collision damp']
        mesh.tex = bitmapIds[mdata.bitmapId]
        mesh.pi = particles.length
        mesh.fluid = mdata['fluid']
        let { scale, offset } = mdata
        let g = await db.transact(db.storeNames, 'readwrite', async tx => await tx.meshGeometry(mid,scale||1,offset||0))
        if (g.verts.length == 0)
            g.verts = [new GeoVert(0, v3(0,0,1))]

        
        for (let vert of g.verts) {
            let p = Particle.alloc()
            p.pos = p.prev_pos = p.rest_pos = vert.pos
            for (let edge of vert.edges)
                p.edges[p.nedges++] = mesh.pi + edge.vert.id
            p.mesh = midx
            p.fixed = mdata.fixed
            particles.push(p)
            
            let adj = vert.ringn(3)
            let c = [0,0,0]
            for (let vadj of adj)
                c = [c[0] + vadj.pos.x, c[1] + vadj.pos.y, c[2] + vadj.pos.z]
            c = c.map(val => val / adj.length)
            p.c0 = v3(...c)
            
            let Q = M3js.of([0,0,0],[0,0,0],[0,0,0])
            for (let vadj of adj) {
                let rx = vadj.pos.x - c[0], ry = vadj.pos.y - c[1], rz = vadj.pos.z - c[2]
                Q = Q.add([[rx*rx, rx*ry, rx*rz], [ry*rx, ry*ry, ry*rz], [rz*rx, rz*ry, rz*rz]])
                p.rings[p.nring++] = vadj.id + mesh.pi
            }

            let s = 1./max(...Q[0], ...Q[1], ...Q[2])
            let Qs = Q.mulc(s)
            p.qinv = M3.of(Qs.invert())
            p.s = s
            
            adj = vert.ringn(4)
            let group = null
            for (let i = 0; i < groups.length && !group; i++)
                if (!adj.some(vadj => groups[i].has(vadj.id + mesh.pi)))
                    group = groups[i]
            if (group) group.add(vert.id + mesh.pi)
            else groups.push(new Set([vert.id + mesh.pi]))
        }
        
        mesh.pf = particles.length - mesh.pi
        for (let face of g.faces) {
            let nverts = face.verts.length
            let ftris = [...range(face.verts.length-2)].map(i => [0,i+1,i+2])
            for (let tri of ftris) {
                let tvs = tri.map(i => TriVert.of(v4(0), face.verts[i].id + mesh.pi, v3(0), midx, v2(...face.uvs[i])))
                tris.push(Triangle.of(...tvs))
            }
        }
        meshes.push(mesh)
    }
    
    if (groups.map(group => group.size).sum() != particles.length)
        throw new Error('bad groups')
    
    meshes = Meshes.of(meshes)
    particles = Particles.of(particles)
    tris = Triangles.of(tris)
    groups = groups.map(group => u32array.of([...group]))
    
    console.timeEnd('db load')

    let longest = 0
    for (let p of particles)
        for (let i of range(p.nedges))
            longest = max(longest, p.pos.dist(particles[p.edges[i]].pos))

    const uniforms = Uniforms.alloc()
    uniforms.selection = uniforms.grabbing = -1
    const lights = GPU.array({ type:Light, length:LIGHTS.length }).alloc(LIGHTS.length)
    for (const [i,l] of enumerate(LIGHTS)) {
        lights[i].color = l.color
        lights[i].pos = l.pos
        lights[i].power = l.power
    }

    dbg({ nparts:particles.length, nmeshes:meshes.length, ntris:tris.length, ngroups:groups.length })

    const threads = gpu.threads

    const bufs = Object.fromEntries([
        gpu.buf({ label:'pbuf', data:particles, usage:'STORAGE|COPY_SRC|COPY_DST|VERTEX' }),
        gpu.buf({ label:'mbuf', data:meshes, usage:'STORAGE|COPY_DST|COPY_SRC' }),
        gpu.buf({ label:'uni', data:uniforms, usage:'UNIFORM|COPY_DST' }),
        gpu.buf({ label:'tribuf', data:tris, usage:'STORAGE|VERTEX|COPY_SRC|COPY_DST' }),
        gpu.buf({ label:'cnts', type:GPU.array({ type:i32, length:threads**3 }), usage:'STORAGE|COPY_DST|COPY_SRC' }),
        gpu.buf({ label:'work1', type:GPU.array({ type:i32, length:threads**2 }), usage:'STORAGE' }),
        gpu.buf({ label:'work2', type:GPU.array({ type:i32, length:threads }), usage:'STORAGE' }),
	gpu.buf({ label:'work3', type:GPU.array({ type:i32, length:1 }), usage:'STORAGE' }),
        gpu.buf({ label:'sorted', type:GPU.array({ type:u32, length:particles.length }), usage:'STORAGE' }),
        gpu.buf({ label:'lbuf', data:lights, usage:'UNIFORM|FRAGMENT|COPY_DST' }),
        gpu.buf({ label:'dbuf', type:GPU.array({ type: f32, length: 16384 }), usage:'STORAGE|COPY_SRC|COPY_DST' }),
    ].map(buf => [buf.label, buf]))

    const pd = ceil(particles.length/gpu.threads)

    const syncUniforms = () => {
        uniforms.r = phys.r * longest/2
        uniforms.cam_pos = render.cam_pos
        uniforms.width = width
        uniforms.height = height
        uniforms.cam_fwd = v3(sin(render.cam_lr) * cos(render.cam_ud),
                              cos(render.cam_lr) * cos(render.cam_ud),
                              sin(render.cam_ud)).normalized()
        uniforms.proj = M4.perspective(render.fov, width/height, .01, 10000)
        uniforms.view = M4.look(render.cam_pos, uniforms.cam_fwd, v3(0,0,1))
        uniforms.mvp = uniforms.view.mul(uniforms.proj)

        phys.keys.filter(k => phys.type[k] == 'num' && k != 'r').forEach(k => {
            uniforms[k] = phys[k]
        })
        uniforms.ground = render.ground ? 1 : 0
        gpu.write(bufs.uni, uniforms)
    }
    
    async function Computer() {

        let fwdstep = false
        let tstart = clock(), tlast = clock()
        let frames = 0, steps = 0

        const threads = gpu.threads
        const wgsl = (await fetchtext('./compute.wgsl')).interp({ threads, MAXNN })

        const shader = await gpu.shader({
            compute:true, wgsl:wgsl, defs:[Particle, Mesh, Uniforms],
            storage:{
                pbuf:Particles, mbuf:Meshes, sorted:u32array, dbuf:f32array,
                cnts:i32array, cnts_atomic:iatomicarray, work:i32array, group:u32array,
                pavg:v3arr, vavg:v3arr, lavg:v3arr, iavg:m3arr,
            },
            uniform:{ uni:Uniforms }
        })

        const { uni, mbuf, pbuf, cnts, work1, work2, work3, sorted, dbuf } = bufs

        const cmds = [
            gpu.timestamp(''),
            gpu.computePass({ pipe:gpu.computePipe({ shader, entryPoint:'predict', binds:['pbuf','mbuf','uni'] }),
                              dispatch:pd, binds:{ pbuf, mbuf, uni } }),
            gpu.timestamp('predict')
        ]
        
        let cntsort_cnt = gpu.computePipe({ shader, entryPoint:'cntsort_cnt', binds:['pbuf','cnts_atomic','uni']})
        let prefsum_down = gpu.computePipe({ shader, entryPoint:'prefsum_down', binds:['cnts','work']})
        let prefsum_up = gpu.computePipe({ shader, entryPoint:'prefsum_up', binds:['cnts','work']})
        let cntsort_sort = gpu.computePipe({ shader, entryPoint:'cntsort_sort', binds:['pbuf','cnts_atomic','sorted']})
        let find_collisions = gpu.computePipe({ shader, entryPoint:'find_collisions', binds:['pbuf','cnts','sorted','uni']})
        let collide = gpu.computePipe({ shader, entryPoint: 'collide', binds:['pbuf','mbuf','uni']})
        let update_pos = gpu.computePipe({ shader, entryPoint:'update_pos', binds:['pbuf'] })
        cmds.push(
            gpu.clearBuffer(cnts, 0, cnts.size),
            gpu.computePass({ pipe:cntsort_cnt, dispatch:pd, binds:{ pbuf, cnts_atomic:cnts, uni }}),
            gpu.timestamp('cntsort_cnt'),
            gpu.computePass({ pipe:prefsum_down, dispatch:threads**2, binds:{ cnts, work:work1 }}),
            gpu.computePass({ pipe:prefsum_down, dispatch:threads, binds:{ cnts:work1, work:work2 }}),
            gpu.computePass({ pipe:prefsum_down, dispatch:1, binds:{ cnts:work2, work:work3 }}),
            gpu.computePass({ pipe:prefsum_up, dispatch:threads - 1, binds:{ cnts:work1, work:work2 }}),
            gpu.computePass({ pipe:prefsum_up, dispatch:threads**2 - 1, binds:{ cnts, work:work1 }}),
            gpu.timestamp('prefsum'),
            gpu.computePass({ pipe:cntsort_sort, dispatch:pd, binds:{ pbuf, cnts_atomic:cnts, sorted }}),
            gpu.timestamp('cntsort_sort'),
            gpu.computePass({ pipe:find_collisions, dispatch:pd, binds:{ pbuf, cnts, sorted, uni }}),
            gpu.timestamp('find collisions'),
            gpu.computePass({ pipe:collide, dispatch:pd, binds:{ pbuf, mbuf, uni } }),
            gpu.timestamp('collide'),
            gpu.computePass({ pipe:update_pos, dispatch:pd, binds:{ pbuf }}),
            gpu.timestamp('update pos')
        )
        
        let pipe = gpu.computePipe({ shader, entryPoint:'surfmatch', binds:['pbuf','mbuf','uni','group','dbuf']})
        for (let i of range(groups.length)) {
            let group = gpu.buf({ label: 'group'+i, data:groups[i], usage:'STORAGE' })
            let dispatch = ceil(groups[i].length / gpu.threads)
            cmds.push(gpu.computePass({ pipe, dispatch, binds:{ pbuf, mbuf, uni, group, dbuf }}))
        }
        cmds.push(gpu.timestamp('surface match'))

        cmds.push(
            gpu.computePass({ pipe:gpu.computePipe({ shader, entryPoint:'update_vel', binds:['pbuf','mbuf','uni'] }),
                              dispatch:pd, binds:{ pbuf, mbuf, uni }}),
            gpu.timestamp('update vel')
        )       

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
        let avgs_prep = gpu.computePipe({ shader, entryPoint:'avgs_prep', binds:['mbuf','pbuf','pavg','vavg','lavg','iavg']})
        let avgs_calc = gpu.computePipe({ shader, entryPoint:'avgs_calc', binds:['mbuf','pavg','vavg','lavg','iavg']})
        
        const meshavgs = (i) => {
            let n = meshes[i].pf - meshes[i].pi
            if (n <= 0) return []
            let dp1 = ceil(n / threads), dp2 = ceil(dp1 / threads)
            cmds.push(gpu.computePass({ pipe:avgs_prep, dispatch:dp1, binds:{ pbuf, ...mbufs[i] }}))
            cmds.push(gpu.computePass({ pipe:avgs_calc, dispatch:dp1, binds:mbufs[i] }))
            if (dp1 > 1) {
                cmds.push(gpu.computePass({ pipe:avgs_calc, dispatch:dp2, binds:mbufs[i] }))
                if (dp2 > 1) cmds.push(gpu.computePass({ pipe:avgs_calc, dispatch:1, binds:mbufs[i]}))
            }
        }

        for (const i of range(meshes.length)) meshavgs(i)
        for (const i of range(meshes.length)) meshavgs(i)
        
        cmds.push(gpu.timestamp('mesh avgs'))
         
        const batch = gpu.encode(cmds)

        return {
            stats: async () => {
                let ret = { kind:'phys', fps:frames / (tlast - tstart) }
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
                    uniforms.dt = tstep
                    uniforms.t += tstep
                    syncUniforms()
                    batch.execute()
                    frames++
                    steps++
                    
                    //gpu.read(pbuf).then(d => { window.ps = new pbuf.type(d) })
                    //gpu.read(dbuf).then(d => { window.dbuf = new dbuf.type(d) })

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
            render.watch(render.keys.filter(key => !['fov','cam_pos','cam_ud','cam_lr'].includes(key)), () => {
                if (render.color_op == 'max' || render.color_op == 'min')
                    render.color_src = render.color_dst = 'one'
                if (render.alpha_op == 'max' || render.alpha_op == 'min')
                    render.alpha_src = render.alpha_dst = 'one'
                reset = true
            })

            const td = ceil(tris.length/gpu.threads)


            let wgsl = (await fetchtext('./prerender.wgsl')).interp({threads: gpu.threads})
            const preShader = await gpu.shader({
                wgsl, compute: true, defs:[ Mesh, Particle, Uniforms, TriVert, Triangle ],
                storage:{  pbuf:Particles, mbuf:Meshes, tribuf:Triangles },
                uniform: { uni:Uniforms }
            })

            wgsl = await fetchtext('./render.wgsl')
            wgsl = wgsl.interp({numLights: lights.length })
            const shader = await gpu.shader({ wgsl, defs:[Particle, Mesh, Uniforms, Light],
                                            storage:{ pbuf:Particles, mbuf:Meshes },
                                            uniform:{ uni:Uniforms, lbuf:lights.constructor },
                                            textures:{ tex:{ name:'texture_2d_array<f32>' } },
                                            samplers:{ samp:{ name:'sampler' } } })

            let tex = gpu.texture(bitmaps)
            let samp = gpu.sampler()

            const { uni, mbuf, pbuf, lbuf, tribuf } = bufs
            
            const cmds = []
            cmds.push(gpu.timestamp(''))
            if (particles.length > 0) {
                const normals = gpu.computePipe({ shader:preShader, entryPoint:'normals', binds: ['pbuf','uni'] })
                cmds.push(gpu.computePass({ pipe:normals, dispatch:pd, binds:{ pbuf, uni } }))
                cmds.push(gpu.timestamp('normals'))
            }
            if (td > 0) {
                const updTris = gpu.computePipe({ shader:preShader, entryPoint:'update_tris', binds: ['tribuf','pbuf'] })
                cmds.push(gpu.computePass({ pipe:updTris, dispatch:td, binds:{ pbuf, tribuf }}))
                cmds.push(gpu.timestamp('updatetris'))
            }

            const draws = []
            if (render.ground) {
                let pipe = gpu.renderPipe({ shader, entry:'ground', cullMode:'none',
                                            binds:['uni','lbuf'], topology:'triangle-strip' })
                draws.push(gpu.draw({ pipe, dispatch:8, binds:{ uni, lbuf }}))
            }

            if (tris.length > 0) {
                let pipe = gpu.renderPipe({ shader, entry:'surface', binds:['mbuf', 'uni', 'lbuf', 'tex', 'samp'],
                    vertBufs: [{ buf:tribuf, arrayStride:TriVert.size,
                                 attributes: [{ shaderLocation:0, offset:TriVert.pos.off, format:'float32x3' },
                                              { shaderLocation:1, offset:TriVert.norm.off, format:'float32x3' },
                                              { shaderLocation:2, offset:TriVert.mesh.off, format:'uint32' },
                                              { shaderLocation:3, offset:TriVert.uv.off, format:'float32x2' }]}]})
                draws.push(gpu.draw({ pipe, dispatch:tris.length*3, binds:{ mbuf, uni, lbuf, tex, samp }}))
            }


            if (render.particles && particles.length > 0) {
                let pipe = gpu.renderPipe({ shader, entry:'particle', binds:['mbuf','uni','lbuf'], topology:'triangle-list',
                    vertBufs: [{ buf:pbuf, arrayStride:Particles.stride, stepMode: 'instance',
                                 attributes: [{ shaderLocation:0, offset:Particle.pos.off, format:'float32x3' },
                                              { shaderLocation:1, offset:Particle.mesh.off, format:'uint32' }]}] })
                draws.push(gpu.draw({ pipe, dispatch:[3, particles.length], binds:{ mbuf, uni, lbuf }}))
            }

            if (render.normals) {
                let pipe = gpu.renderPipe({ shader, entry:'vnormals', binds: ['uni'], topology: 'line-list',
                    vertBufs: [{ buf:tribuf, arrayStride:TriVert.size, stepMode: 'instance',
                    attributes: [{ shaderLocation:0, offset:TriVert.pos.off, format:'float32x3' },
                                 { shaderLocation:1, offset:TriVert.norm.off, format:'float32x3' }]}]})
                draws.push(gpu.draw({ pipe, dispatch:[2, tris.length*3], binds:{ uni }}))
            }

            
            let pipe = gpu.renderPipe({ shader, entry:'lights', binds: ['uni','lbuf'], topology: 'triangle-list',
                                        atc:false, depthWriteEnabled:false })
            draws.push(gpu.draw({ pipe, dispatch:[3, lights.length], binds: { uni, lbuf }}))
            cmds.push(gpu.renderPass(draws))
            cmds.push(gpu.timestamp('draws'))

            batch = gpu.encode(cmds)
        }

        await setup()

        return {
            stats: async () => {
                const ret = { kind:'render', fps: frames/(tlast - tstart) }
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

    const computer = await Computer()
    const renderer = await Renderer()

    const pull = async (bufName) => {
        return new bufs[bufName].type(await gpu.read(bufs[bufName]))
    }

    const clipToRay = (x,y) => {
        let clip = v4(2*x/width - 1, 1 - 2*y/height,-1,1)
        let eye = uniforms.proj.inverse().transform(clip)
        let ray = uniforms.view.inverse().transform(v4(eye.x,eye.y,-1,0))
        return v3(ray.x,ray.y,ray.z).normalized()
    }

    function resize(w, h) {
        width = w
        height = h
        dbg({width,height})
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
        dbg({done:true})
    }

    function rotateCam(dx, dy) {
        render.cam_lr += dx
        render.cam_ud = clamp(render.cam_ud + dy, -PI / 2, PI / 2)
    }

    function strafeCam(dx, dy) {
        const delta = v3(-dx * cos(render.cam_lr), dx * sin(render.cam_lr), -dy)
        render.cam_pos = render.cam_pos.add(delta)
    }

    function advanceCam(delta) {
        const camDir = v3(sin(render.cam_lr) * cos(render.cam_ud),
                          cos(render.cam_lr) * cos(render.cam_ud),
                          sin(render.cam_ud)).normalized()
        render.cam_pos = render.cam_pos.add(v3(delta * camDir.x, delta * camDir.y, delta * camDir.z))
    }

    let grabDepth

    async function grabParticle(x, y) {
        let ray = clipToRay(x,y)
        let rsq = (uniforms.r)**2
        let particles = new Particles(await gpu.read(bufs.pbuf))

        let hitdists = []
        for (const [i, p] of enumerate(particles)) {
            let co = render.cam_pos.sub(p.pos)
            let b = ray.dot(co)
            let discrim = b*b - co.dot(co) + rsq
            if (discrim < 0) continue
            let dist = -b - sqrt(discrim)
            if (dist > 0) hitdists.push([i,dist])
        }
        uniforms.selection = uniforms.grabbing = -1
        if (hitdists.length == 0) return
        hitdists.sort((a,b) => a[1]-b[1])
        uniforms.selection = hitdists[0][0]
        let p = particles[uniforms.selection]
        dbg({pid:uniforms.selection, pos:p.pos, norm:p.norm})
        let fwd = v3(sin(render.cam_lr), cos(render.cam_lr), 0).normalized()
        grabDepth = particles[uniforms.selection].pos.sub(render.cam_pos).dot(fwd)
    }

    function moveParticle(x, y, drop) {
        if (uniforms.selection == -1) return
        let r = clipToRay(x,y)
        let fwd = v3(sin(render.cam_lr), cos(render.cam_lr), 0).normalized()
        let c = grabDepth/r.dot(fwd)
        let R = r.mulc(c)
        uniforms.grabTarget = render.cam_pos.add(R)
        uniforms.grabbing = drop ? -1 : uniforms.selection
    }

    async function fixParticle() {
        let pid = uniforms.selection
        if (pid < 0) return
        let buf = gpu.chop(gpu.offset(bufs.pbuf, Particles.stride * pid), Particle.size)
        let p = new Particle(await gpu.read(buf))
        p.fixed = p.fixed == 1 ? 0 : 1
        dbg({fixed:p.fixed})
        gpu.write(buf, p)
    }

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
            let pos = vdata.slice(0, 3)
            vertIds[localVerts++] = await transaction.wait(verts.add({ pos, meshId }))
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

export const loadBitmap = async (name, data, transaction) => {
    const img = new Image()
    img.src = data
    await img.decode()
    const bitmap = await createImageBitmap(img)
    let bitmaps = transaction.objectStore('bitmaps')
    await transaction.wait(bitmaps.add({ name, data: bitmap }))
}






