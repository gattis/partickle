import * as util from './utils.js'
import * as gpu from './gpu.js'
import * as geo from './geometry.js'
Object.assign(globalThis, util, gpu, geo)

const MAXEDGES = 18
const RINGITER = 2
const MAXRING = 48
const CELLCAP = 8
const NHASH = 128**3

const particleColors = [v4(.3,.6,.8,1), v4(.99,.44,.57, 1), v4(.9, .48, .48, 1)]
const LIGHTS = [
    { power:1.5, color:v3(1,.85,.6), x:v3(2,2,2.3) },
    { power:1.5, color:v3(1,.85,.6), x:v3(2,-2,2.3) },
    { power:1.5, color:v3(1,.85,.6), x:v3(-2,2,2.3) },
    { power:1.5, color:v3(1,.85,.6), x:v3(-2,-2,2.3) },
]

const MESH_DEFAULTS = {
    name:'default', bitmapId:-1, color:[1,1,1,1], offset:[0,0,0], rotation:[0,0,0],
    scale:[1,1,1], volstiff:1, shearstiff:1, friction:1, collision:1, fluid:0, fixed: 0
}

export const phys = new Preferences('phys')
phys.addBool('paused', false)
phys.addNum('r', 1.0, 0.1, 10, 0.1)
phys.addNum('frameratio', 3, 1, 20, 1)
phys.addNum('speed', 1, 0.05, 5, .01)
phys.addNum('gravity', 9.8, -5, 20, 0.1)
phys.addNum('volstiff', .5, 0, 1, 0.001)
phys.addNum('shearstiff', .5, 0, 1, 0.001)
phys.addNum('damp', 0.5, -100, 100, .1)
phys.addNum('friction', 1, 0, 1, .01)
phys.addNum('collision', 1, 0, 2, .01)
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
        ['hash', i32],
        ['x0', V3],
        ['mesh', u32],
        ['xprev', V3],
        ['fixed', i32],
        ['v', V3],
        ['nedges', u32],
        ['norm', V3],
        ['nring', u32],        
        ['c0', V3],
        ['s', f32],
        ['pmin',V3],
        ['friction',f32],
        ['pmax',V3],
        ['collision', f32],
        ['dx', V3],
        ['ndx', f32],
        ['grab', i32],
        ['cellpos', i32],
        ['qinv', M3],
        ['edges', GPU.array({ type:u32, length:MAXEDGES })],
        ['rings', GPU.array({ type:u32, length:MAXRING })]
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
        ['grabbing', i32],
        ['grabStart', i32],
        ['grabRay', V3],
        ['grabTarget', V3],
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

export const Bounds = GPU.struct({
    name:'Bounds',
    fields:[
        ['min', V3],
        ['max', V3],        
        ['grid', V3I],
        ['stride', V3I],
    ]
})

export const Hit = GPU.struct({
    name:'Hit',
    fields:[
        ['pid', u32],
        ['x', V3]
    ]
})

export const HitList = GPU.struct({
    name:'HitList',
    fields:[
        ['list', GPU.array({ type:Hit, length:8192 })],
        ['len', iatomic]
    ]
})

export const Cell = GPU.struct({
    name:'Cell',
    fields:[
        ['pids', GPU.array({ type:u32, length:8 })],
        ['npids', u32],
        ['adj', GPU.array({ type:i32, length:13 })],
    ]
})
            


const Meshes = GPU.array({ type:Mesh })
const Particles = GPU.array({ type:Particle })
const Triangles = GPU.array({ type:Triangle })
const Cells = GPU.array({ type:Cell })

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
        
    let meshes = [], particles = [], tris = [], groups = [], groups_exclude = []
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
            for (let edge of vert.edges)
                p.edges[p.nedges++] = mesh.pi + edge.vert.id
            p.mesh = midx
            p.fixed = mdata.fixed
            p.friction = mdata.friction
            p.collision = mdata.collision
            particles.push(p)           

            let adj = vert.ringn(RINGITER)
            let c = [0,0,0]
            for (let vadj of adj.verts)
                c = [c[0] + vadj.x.x, c[1] + vadj.x.y, c[2] + vadj.x.z]
            c = c.map(val => val / adj.verts.length)
            p.c0 = v3(...c)
            
            let Q = M3js.of([0,0,0],[0,0,0],[0,0,0])
            for (let vadj of adj.verts) {
                let rx = vadj.x.x - c[0], ry = vadj.x.y - c[1], rz = vadj.x.z - c[2]
                Q = Q.add([[rx*rx, rx*ry, rx*rz], [ry*rx, ry*ry, ry*rz], [rz*rx, rz*ry, rz*rz]])
                p.rings[p.nring++] = vadj.id + mesh.pi
            }

            let qsum = Q[0].sum() + Q[1].sum() + Q[2].sum()
            if (qsum > 0) {
                let s = 1/qsum
                let Qs = Q.mulc(s)
                p.qinv = M3.of(Qs.invert())
                p.s = s
            }
            
            adj = vert.ringn(RINGITER+1)
            let group = -1
            for (let i = 0; i < groups.length && group == -1; i++)
                if (!adj.verts.some(vadj => groups_exclude[i].has(vadj.id + mesh.pi)))
                    group = i
            if (group == -1) {
                group = groups.length
                groups.push([])
                groups_exclude.push(new Set())                            
            } 
            groups[group].push(vert.id + mesh.pi)
            for (let vadj of adj.verts)
                groups_exclude[group].add(vadj.id + mesh.pi)           

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
    tris = Triangles.of(tris)
    groups = groups.map(group => u32arr.of([...group]))

    let diameter = -1
    for (let p of particles)
        for (let i of range(p.nedges))
            diameter = max(diameter, p.x.dist(particles[p.edges[i]].x))
    if (diameter < 0) diameter = 0.02

    const uniforms = Uniforms.alloc()
    uniforms.seed = 666
    uniforms.selection = uniforms.grabbing = -1

    const lights = GPU.array({ type:Light, length:LIGHTS.length }).alloc(LIGHTS.length)
    for (const [i,l] of enumerate(LIGHTS)) {
        lights[i].color = l.color
        lights[i].x = l.x
        lights[i].power = l.power
    }
    
    const threads = gpu.threads
    let hitList = HitList.alloc()
    
    const bufs = Object.fromEntries([
        gpu.buf({ label:'pbuf', data:particles, usage:'STORAGE|COPY_SRC|COPY_DST|VERTEX' }),
        gpu.buf({ label:'mbuf', data:meshes, usage:'STORAGE|COPY_DST|COPY_SRC' }),
        gpu.buf({ label:'uni', data:uniforms, usage:'UNIFORM|COPY_DST' }),
        gpu.buf({ label:'tribuf', data:tris, usage:'STORAGE|VERTEX|COPY_DST' }),
        gpu.buf({ label:'lbuf', data:lights, usage:'UNIFORM|FRAGMENT|COPY_DST' }),
        gpu.buf({ label:'hitlist', data:hitList, usage:'STORAGE|COPY_SRC|COPY_DST' })
    ].map(buf => [buf.label, buf]))

    const pd = ceil(particles.length/gpu.threads), pd2 = ceil(pd / gpu.threads)

    console.timeEnd('db load')
    
    const syncUniforms = () => {
        uniforms.r = phys.r * diameter/2
        uniforms.cam_x = render.cam_x
        uniforms.width = width
        uniforms.height = height
        uniforms.cam_fwd = v3(sin(render.cam_lr) * cos(render.cam_ud),
                              cos(render.cam_lr) * cos(render.cam_ud),
                              sin(render.cam_ud)).normalized()
        uniforms.proj = M4.perspective(render.fov, width/height, .001, 100)
        uniforms.view = M4.look(render.cam_x, uniforms.cam_fwd, v3(0,0,1))
        uniforms.mvp = uniforms.view.mul(uniforms.proj)
        uniforms.spacemin = v3(phys.xmin, phys.ymin, phys.zmin)
        uniforms.spacemax = v3(phys.xmax, phys.ymax, phys.zmax)
        uniforms.damp = phys.damp
        uniforms.a = v3(0,0,-phys.gravity);
        uniforms.friction = phys.friction
        uniforms.collision = phys.collision
        uniforms.volstiff = phys.volstiff
        uniforms.shearstiff = phys.shearstiff

        return gpu.write(bufs.uni, uniforms)
    }
    
    async function Computer() {
        
        let fwdstep = false
        let tstart = clock(), tlast = clock()
        let frames = 0, steps = 0

        const threads = gpu.threads
        const wgsl = (await fetchtext('./compute.wgsl')).interp({ threads, NHASH, CELLCAP })

        let csAlign = gpu.adapter.limits.minStorageBufferOffsetAlignment/4
        let csCap = ceil(particles.length/csAlign)*csAlign
        let CellSet = GPU.array({ type:i32, length:csCap })
        let CellSets = GPU.array({ type: CellSet, length:3**3 })
        let SetLens = GPU.array({ type:GPU.array({ type:uatomic, length:csAlign }), length:3**3 })


        const shader = await gpu.shader({
            compute:true, wgsl:wgsl, defs:[Particle, Mesh, Uniforms, Bounds, Hit, HitList, Cell],
            storage:{
                pbuf:Particles, mbuf:Meshes, hitlist:HitList, setdps:v3uarr, cells:Cells, idp:V3U,
                cnts:iatomicarr, group:u32arr, cellsets:CellSets, cellset:CellSet, ncells:uatomic,
                pavg:v3arr, vavg:v3arr, lavg:v3arr, iavg:m3arr, bounds:Bounds, setlens:SetLens, setlen:u32arr
            },
            uniform:{ uni:Uniforms }
        })

        const { uni, mbuf, pbuf } = bufs

        const cmds = []
        const pass = (...args) => { cmds.push(gpu.computePass(...args)) }
        const pipe = (...args) => gpu.computePipe(...args)
        const stamp = (tag) => { cmds.push(gpu.timestamp(tag)) }
        let keys = obj => [...Object.keys(obj)]
        
        let bounds = gpu.buf({ label:'bounds', data:Bounds.alloc(), usage:'STORAGE|COPY_SRC' })
        let cells = gpu.buf({ label:'cells', type:GPU.array({type:Cell, length:particles.length}), usage:'STORAGE|COPY_DST|COPY_SRC' })
        let ncells = gpu.buf({ label:'ncells', type:uatomic, usage:'STORAGE|COPY_SRC' })
        let cellsets = gpu.buf({ label:'cellsets', type:CellSets, usage:'STORAGE|COPY_SRC'})
        let setlens = gpu.buf({ label:'setlens', type:SetLens, usage:'STORAGE|COPY_SRC'})
        let cnts = gpu.buf({ label:'cnts', type:GPU.array({ type:i32, length:NHASH }), usage:'STORAGE|COPY_DST' })
        let idp = gpu.buf({ label:'idp', type:V3U, usage:'STORAGE|INDIRECT|COPY_SRC' })
        let setdps = gpu.buf({ label:'setdps', type:GPU.array({ type:V3U, length:3**3 }), usage:'STORAGE|INDIRECT|COPY_SRC'})

        stamp('')

        let binds = { pbuf, mbuf, uni, hitlist:bufs.hitlist }
        pass({ pipe:pipe({ shader, entryPoint:'predict', binds:keys(binds) }), dispatch:pd, binds })
        stamp('predict')

        binds = { pbuf, bounds, setlens, uni, ncells }
        let find_bounds = pipe({ shader, entryPoint:'find_bounds', binds:keys(binds)})
        pass({ pipe:find_bounds, dispatch:pd, binds })
        if (pd > 1) {
            pass({ pipe:find_bounds, dispatch:pd2, binds })
            if (pd2 > 1) pass({ pipe:find_bounds, dispatch:1, binds })
        }
               
        cmds.push(gpu.clearBuffer(cnts, 0, cnts.size))

        binds = { pbuf, cnts, uni, bounds, cells }
        pass({ pipe:pipe({ shader, entryPoint:'cellcount', binds:keys(binds) }), dispatch:pd, binds })
        binds = { pbuf, cnts, uni, bounds, cells, setlens, cellsets, ncells }
        pass({ pipe:pipe({ shader, entryPoint:'initcells', binds:keys(binds)}), dispatch:pd, binds }) 
        binds = { pbuf, cnts, uni, bounds, cells, setlens, setdps, idp, ncells }
        pass({ pipe:pipe({ shader, entryPoint:'fillcells', binds:keys(binds) }), dispatch:pd, binds })
        stamp('cell grid build')

        binds = { pbuf, uni, cells, ncells }
        pass({ pipe:pipe({ shader, entryPoint:'intercell', binds:keys(binds) }), indirect:[idp.buffer, 0], binds })
        stamp('intercell collisions')

        binds = { pbuf, uni, cells }
        let intracell = pipe({ shader, entryPoint:'intracell', binds:[...keys(binds), 'setlen','cellset'] })
        for (let i of range(3**3))
            pass({ pipe:intracell, indirect:[setdps.buffer, i*v3uarr.stride], binds:{ ...binds,
                   cellset:gpu.offset(cellsets, i*CellSets.stride), setlen:gpu.offset(setlens, i*SetLens.stride) }})
        stamp('intracell collisions')

        binds = { pbuf, mbuf, uni }
        pass({ pipe:pipe({ shader, entryPoint:'collide_bounds', binds:keys(binds) }), dispatch:pd, binds })
        stamp('bounds collide')       
        
        binds = { pbuf, mbuf, uni }
        let surfmatch = pipe({ shader, entryPoint:'surfmatch', binds:[...keys(binds), 'group']})
        for (let i of range(groups.length))
            pass({ pipe:surfmatch, dispatch:ceil(groups[i].length / threads),
                   binds:{ ...binds, group:gpu.buf({ label: 'group'+i, data:groups[i], usage:'STORAGE' })}})
        stamp('surface match')

        binds = { pbuf, mbuf, uni }
        pass({ pipe:pipe({ shader, entryPoint:'update_vel', binds:keys(binds) }), dispatch:pd, binds })
        stamp('update vel')

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

        return {
            stats: async () => {
                let ret = { kind:'phys', fps:frames / (tlast - tstart) * phys.speed }
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
                    uniforms.seed = uniforms.seed * 16807 % 2147483647
                    syncUniforms()
                    batch.execute()
                    if (uniforms.grabStart == 1) {
                        uniforms.grabStart = 0
                        uniforms.selection = -1
                        hitList = new HitList(await gpu.read(bufs.hitlist))
                        if (hitList.len == 0) return
                        let closest = { dist:Infinity }
                        for (let i of range(hitList.len)) {
                            let dist = hitList.list[i].x.sub(render.cam_x).mag()
                            if (dist < closest.dist) closest = {i, dist}
                        }                        
                        let hit = hitList.list[closest.i]
                        uniforms.selection = uniforms.grabbing = hit.pid
                        uniforms.grabTarget = hit.x.clamp(uniforms.spacemin.addc(uniforms.r), uniforms.spacemax)
                        let fwd = v3(sin(render.cam_lr), cos(render.cam_lr), 0).normalized()
                        grabDepth = hit.x.sub(render.cam_x).dot(fwd)
                        hitList.len = 0
                        gpu.write(bufs.hitlist, hitList)
                        dbg({ pid:uniforms.selection, x:hit.x, dist:closest.dist })
                    }
                                            
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
            if (render.walls) {
                let pipe = gpu.renderPipe({ shader, entry:'walls', cullMode:'back',
                                            binds:['uni','lbuf'], topology:'triangle-strip' })
                draws.push(gpu.draw({ pipe, dispatch:14, binds:{ uni, lbuf }}))
            }

            if (tris.length > 0) {
                let pipe = gpu.renderPipe({ shader, entry:'surface', binds:['mbuf', 'uni', 'lbuf', 'tex', 'samp'],
                    vertBufs: [{ buf:tribuf, arrayStride:TriVert.size,
                                 attributes: [{ shaderLocation:0, offset:TriVert.x.off, format:'float32x3' },
                                              { shaderLocation:1, offset:TriVert.norm.off, format:'float32x3' },
                                              { shaderLocation:2, offset:TriVert.mesh.off, format:'uint32' },
                                              { shaderLocation:3, offset:TriVert.uv.off, format:'float32x2' }]}]})
                draws.push(gpu.draw({ pipe, dispatch:tris.length*3, binds:{ mbuf, uni, lbuf, tex, samp }}))
            }

            if (render.particles && particles.length > 0) {
                let pipe = gpu.renderPipe({ shader, entry:'particle', binds:['mbuf','uni','lbuf'], topology:'triangle-list',
                    vertBufs: [{ buf:pbuf, arrayStride:Particles.stride, stepMode: 'instance',
                                 attributes: [{ shaderLocation:0, offset:Particle.x.off, format:'float32x3' },
                                              { shaderLocation:1, offset:Particle.mesh.off, format:'uint32' }]}] })
                draws.push(gpu.draw({ pipe, dispatch:[3, particles.length], binds:{ mbuf, uni, lbuf }}))
            }

            if (render.normals) {
                let pipe = gpu.renderPipe({ shader, entry:'vnormals', binds: ['uni'], topology: 'line-list',
                    vertBufs: [{ buf:tribuf, arrayStride:TriVert.size, stepMode: 'instance',
                    attributes: [{ shaderLocation:0, offset:TriVert.x.off, format:'float32x3' },
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
                const ret = { kind:'render', fps: frames / (tlast - tstart) * phys.speed }
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
        //dbg(Object.fromEntries([[buf.label, data]]))
        return data
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
        render.cam_x = render.cam_x.add(delta)
    }

    function advanceCam(delta) {
        const camDir = v3(sin(render.cam_lr) * cos(render.cam_ud),
                          cos(render.cam_lr) * cos(render.cam_ud),
                          sin(render.cam_ud)).normalized()
        render.cam_x = render.cam_x.add(v3(delta * camDir.x, delta * camDir.y, delta * camDir.z))
    }

    let grabDepth

    async function grabParticle(x, y) {
        uniforms.grabRay = clipToRay(x,y)
        uniforms.grabStart = 1
        uniforms.grabbing = -1
    }

    function moveParticle(x, y) {
        if (uniforms.grabbing == -1) return
        let r = clipToRay(x,y)
        let fwd = v3(sin(render.cam_lr), cos(render.cam_lr), 0).normalized()
        let c = grabDepth/r.dot(fwd)
        let R = r.mulc(c)
        uniforms.grabTarget = render.cam_x.add(R)
        uniforms.grabTarget = uniforms.grabTarget.max(uniforms.spacemin.addc(uniforms.r))
        uniforms.grabTarget = uniforms.grabTarget.min(uniforms.spacemax.subc(uniforms.r))
    }

    function dropParticle() {
        uniforms.grabbing = -1
    }

    async function fixParticle() {
        let pid = uniforms.selection
        if (pid < 0) return
        dbg({fixing:pid})
        let buf = gpu.chop(gpu.offset(bufs.pbuf, Particles.stride * pid), Particle.size)
        let p = new Particle(await gpu.read(buf))
        p.fixed = p.fixed == 1 ? 0 : 1
        dbg({fixed:p.fixed})
        gpu.write(buf, p)
    }

    console.time('compute init')
    const computer = await Computer()
    console.timeEnd('compute init')
    console.time('renderer init')
    const renderer = await Renderer()
    console.timeEnd('renderer init')

    dbg({ nparts:particles.length, nmeshes:meshes.length, ntris:tris.length, ngroups:groups.length, r:diameter/2 })
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






