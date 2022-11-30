import * as util from './utils.js'
import * as gpu from './gpu.js'
import * as geo from './geometry.js'
Object.assign(globalThis, util, gpu, geo)

const MAXNN = 18
const MAXEDGES = 18
const TETLIM = sqrt(2)
let UP = v3(0,0,1)
let CAM_POS = v3(0, -3.5, 3)
let CAM_UD = -PI/6
const particleColors = [v4(.3,.6,.8,1), v4(.99,.44,.57, 1), v4(.9, .48, .48, 1)]
const LIGHTS = [
    { power:1.5, color:v3(1,.85,.6), pos:v3(2,2,2.3) },
    { power:1.5, color:v3(1,.85,.6), pos:v3(2,-2,2.3) },
    { power:1.5, color:v3(1,.85,.6), pos:v3(-2,2,2.3) },
    { power:1.5, color:v3(1,.85,.6), pos:v3(-2,-2,2.3) },
]

export const phys = new Preferences('phys')
phys.addBool('paused', false)
phys.addNum('r', .01, 0, 0.1, 0.001)
phys.addNum('density', 1.0, 0.1, 10, 0.01)
phys.addNum('frameratio', 3, 1, 20, 1)
phys.addNum('speed', 1, 0.05, 5, .01)
phys.addNum('gravity', 9.8, -5, 20, 0.01)
phys.addNum('shape_stiff', 0, 0, 1, 0.005)
phys.addNum('vol_stiff', .5, 0, 1, 0.005)
phys.addNum('damp', 0.5, 0, 1, .001)
phys.addNum('collidamp', .1, 0, 1, .001)


export const render = new Preferences('render')
render.addBool('particles', true)
render.addBool('ground', true)
render.addBool('normals', false)
render.addBool('edges', false)
render.addBool('depth_wr', true)
render.addBool('atc', true)
render.addChoice('depth_cmp', 'less', ['less-equal','less','greater-equal','greater','always','never'])
render.addChoice('cull', 'back', ['none','back','front'])
render.addChoice('alpha_mode', 'premultiplied', ['opaque','premultiplied'])
render.addChoice('color_op', 'add', ['add','subtract','reverse-subtract'])
render.addChoice('alpha_op', 'add', ['add','subtract','reverse-subtract'])
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

const clock = () => phys.speed*performance.now()/1000

export const Mesh = GPU.struct({
    name:'Mesh',
    fields:[
        ['c0', V3],
        ['ci', V3],
        ['pi', u32],
        ['pf', u32],
        ['vi', u32],
        ['vf', u32],
        ['rot', M3],
        ['tex', i32],
        ['color', V4],        
        ['pcolor', V4],
        ['gravity', f32],
        ['shape_stiff', f32],
        ['vol_stiff', f32],
        ['friction', f32],
        ['collidamp', f32],
        ['fluid', i32],
        ['pose', i32],
        ['inactive', i32],
        ['padding', GPU.array({ type:u32, length:20 })]
    ]
})

export const Vertex = GPU.struct({
    name:'Vertex',
    fields:[
        ['pos', V3],
        ['mesh', u32],
        ['q', V3],
        ['bary', V3],
        ['tet', i32],
        ['norm', V3],
        ['nedges', u32],
        ['edges', GPU.array({ type:u32, length:MAXEDGES })],
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
        ['w', f32],
        ['vel', V3],
        ['k', u32],
        ['q', V3],
        ['delta_pos', V3],
        ['norm', V3],
        ['nn', GPU.array({ type:u32, length:MAXNN })],
    ]
})


export const Camera = GPU.struct({
    name:'Camera',
    fields:[
        ['projection', M4],
        ['modelview', M4],
        ['inverse', M4],
        ['pos', V3],
        ['selection', i32],
        ['forward', V3],
        ['r', f32],
        ['ratio', f32],
        ['t', f32]
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

export const Params = GPU.struct({
    name:'Params',
    fields:[
        ...phys.keys.filter(k => phys.type[k] == 'num').map(k => [k, f32]),
        ['handpos', V3],
        ['handrot', M3],
        ['ground', u32],
        ['t',f32],
        ['step', u32],
    ]
})

export const TriVert = GPU.struct({
    name:'TriVert',
    fields:[
        ['pos', V3],
        ['vidx', u32],
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

export const Edge = GPU.array({ type: u32, length: 2 })
export const Tet = GPU.array({ type: u32, length: 4 })

const Meshes = GPU.array({ type:Mesh })
const Vertices = GPU.array({ type:Vertex })
const Particles = GPU.array({ type:Particle })
const Triangles = GPU.array({ type:Triangle })
const Tets = GPU.array({ type:Tet })
const Edges = GPU.array({ type:Edge })


export class Sim {

    async init(width, height, ctx, db) {
        
        const gpu = new GPU()
        await gpu.init(width,height,ctx)
        this.camPos = CAM_POS
        this.camLR = 0
        this.camUD = CAM_UD
        this.handPos = v3(0, 0, 0)
        this.handRot = m3([[1,0,0],[0,1,0],[0,0,1]])
        this.handUD = 0

        let bitmapIds = {'-1':-1}
        
        console.time('db load')
        let [bitmapData, meshData, caches] = await db.transact(['bitmaps','meshes','cache'],'readonly', 
            async x => await Promise.all([x.query('bitmaps'), x.query('meshes'), x.query('cache')]))
        
        let bitmaps = []
        for (let [bidx,[bid,bdata]] of enumerate(bitmapData)) {
            bitmapIds[bid] = bidx
            bitmaps.push(bdata.data)
        }
        
        meshData = [...meshData]
        for (let [mid,mdata] of meshData) {
            let cache = caches.get(mid)
            if (cache == undefined)
                cache = await db.transact(db.storeNames, 'readwrite', async x => await sampleMesh(mid, 2*phys.r, x))
            mdata.ppos = new v3array(cache.particles)
            mdata.tets = new Tets(cache.tets)
            mdata.mass = new Float32Array(cache.mass)
            mdata.verts = new Vertices(cache.verts)
            mdata.faces = new Triangles(cache.faces)
            mdata.edges = new u32array(cache.edges)
            mdata.tetGroups = cache.tetGroups
        }

        let meshes = Meshes.alloc(meshData.length)
        let particles = Particles.alloc(meshData.map(([,mdata]) => mdata.ppos.length).sum())
        let verts = Vertices.alloc(meshData.map(([,mdata]) => mdata.verts.length).sum())
        let tris = Triangles.alloc(meshData.map(([,mdata]) => mdata.faces.length).sum())
        let tets = Tets.alloc(meshData.map(([,mdata]) => mdata.tets.length).sum())
        let edges = u32array.alloc(meshData.map(([,mdata]) => mdata.edges.length).sum())
        let tetGroups = []

        let nparticles = 0, ntets = 0, nverts = 0, ntris = 0, nedges = 0
        for (let [midx,[mid,mdata]] of enumerate(meshData)) {
            let mesh = meshes[midx]
            mesh.color = v4(...mdata.color)
            mesh.pcolor = particleColors[midx % particleColors.length]
            mesh.gravity = mdata.gravity
            mesh.shape_stiff = mdata['shape stiff']
            mesh.vol_stiff = mdata['vol stiff']
            mesh.friction = mdata.friction
            mesh.collidamp = mdata['collision damp']
            mesh.fluid = mdata['fluid']
            mesh.tex = bitmapIds[mdata.bitmapId]
            mesh.rot = m3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            mesh.pi = nparticles
            mesh.pf = nparticles + mdata.ppos.length
            mesh.vi = nverts
            mesh.vf = nverts + mdata.verts.length
            let ti = ntets
            for (let pidx of range(mdata.ppos.length))  {
                let p = particles[nparticles++]
                p.pos = p.prev_pos = p.rest_pos = mdata.ppos[pidx]
                mesh.c0 = mesh.c0.add(p.pos)
                p.mesh = midx
                p.w = 1.0 / mdata.mass[pidx] * mdata.invmass
            }
            mesh.c0 = mesh.ci = mesh.c0.divc(mesh.pf - mesh.pi)
            for (let pidx of range(mdata.ppos.length))  {
                let p = particles[mesh.pi + pidx]
                p.q = p.pos.sub(mesh.c0)
            }
            for (let [tidx,tet] of enumerate(mdata.tets))
                tets[ntets++] = Tet.of([0,1,2,3].map(i => tet[i] + mesh.pi))
            for (let tri of mdata.faces) {
                for (let i of range(3)) {
                    tri[i].vidx += nverts
                    tri[i].mesh = midx
                }
                tris[ntris++] = tri
            }
            for (const vidx of range(mdata.verts.length)) {
                let v = mdata.verts[vidx]
                v.mesh = midx
                for (let i of range(v.nedges))
                    v.edges[i] += mesh.vi
                v.tet += ti
                v.q = v.pos.sub(mesh.c0)
                verts[nverts++] = v
            }
            for (const pidx of mdata.edges)
                edges[nedges++] = pidx + mesh.pi
            for (let [i,meshGroup] of enumerate(mdata.tetGroups)) {
                if (i >= tetGroups.length) tetGroups.push([])
                let group = tetGroups[i]
                for (let tid of meshGroup)
                    group.push(tid + ti)
            }
        }
        console.timeEnd('db load')

        tetGroups = tetGroups.map((group,i) => ({
            data:group, 
            buf:gpu.buf({ label:`tetgroup${i}`, data:u32array.of(group), usage:'STORAGE' })
        }))

        const params = Params.alloc()

        const camera = Camera.alloc()
        const lights = GPU.array({ type:Light, length:LIGHTS.length }).alloc(LIGHTS.length)
        for (const [i,l] of enumerate(LIGHTS)) {
            lights[i].color = l.color
            lights[i].pos = l.pos
            lights[i].power = l.power
        }
        
        dbg({ nparts:particles.length, nverts:verts.length, nmeshes:meshes.length, ntets:tets.length, ntris:tris.length })
        const threads = gpu.threads
        
        const buflist = [
            gpu.buf({ label:'particles', data:particles, usage:'STORAGE|COPY_SRC|COPY_DST|VERTEX' }),
            gpu.buf({ label:'tets', data:tets, usage:'STORAGE|COPY_DST|COPY_SRC' }),
            gpu.buf({ label:'edges', data:edges, usage:'STORAGE|COPY_DST|COPY_SRC|INDEX' }),
            gpu.buf({ label:'vertices', data:verts, usage:'STORAGE|VERTEX|COPY_SRC' }),
            gpu.buf({ label:'meshes', data:meshes, usage:'STORAGE|COPY_DST|COPY_SRC' }),
            gpu.buf({ label:'camera', data:camera, usage:'UNIFORM|COPY_DST' }),
            gpu.buf({ label:'params', data:params, usage:'UNIFORM|COPY_DST' }),
            gpu.buf({ label:'tris', data:tris, usage:'STORAGE|VERTEX|COPY_SRC|COPY_DST' }),
            gpu.buf({ label:'cnts', type:GPU.array({ type:i32, length:threads**3 }), usage:'STORAGE|COPY_DST|COPY_SRC' }),
            gpu.buf({ label:'work1', type:GPU.array({ type:i32, length:threads**2 }), usage:'STORAGE' }),
            gpu.buf({ label:'work2', type:GPU.array({ type:i32, length:threads }), usage:'STORAGE' }),
            gpu.buf({ label:'sorted', type:GPU.array({ type:u32, length:particles.length}), usage:'STORAGE' }),
            gpu.buf({ label:'lights', data:lights, usage:'UNIFORM|FRAGMENT|COPY_DST' }),
            gpu.buf({ label:'debug', type:GPU.array({ type: f32, length:4096 }), usage:'STORAGE|COPY_SRC|COPY_DST' }),
            
        ]


        const bufs = Object.fromEntries(buflist.map(buf => [buf.label,buf]))
        Object.assign(this, {gpu, meshes, verts, particles, tris, tets, bitmaps, camera, lights, bufs,
                             ctx, width, height, params, edges, tetGroups,fixed:{} })

        this.compute = new Compute(this)
        this.render = new Render(this)
        
        await this.compute.setup()
        await this.render.setup()
        return this
    }

    resize(width, height) {
        this.width = width
        this.height = height
        this.gpu.configure(this.ctx,width,height,render)
    }

    camFwd() {
        return v3(sin(this.camLR) * cos(this.camUD), cos(this.camLR) * cos(this.camUD), sin(this.camUD)).normalized()
    }

    camAxes() {
        let fwd = this.camFwd().normalized()
        let right = fwd.cross(v3(0,0,1)).normalized()
        let up = right.cross(fwd).normalized()
        return { fwd, up, right }
    }

    handFwd() {
        return v3(sin(this.camLR) * cos(this.handUD), cos(this.camLR) * cos(this.handUD), sin(this.handUD)).normalized()
    }

    async run() {
        const { compute, render } = this
        while (!this.stopRequest) {
            const p = new Promise(resolve => requestAnimationFrame(resolve))
            const tstart = clock()
            await render.step()
            for (let i of range(phys.frameratio))
                await compute.step(1/60/phys.frameratio * phys.speed)
            await p
        }
        this.stopRequest()
        delete this.stopRequest 
    }

    async stop() {
        if (this.stopRequest) return
        await new Promise(resolve => { this.stopRequest = resolve })
        dbg('destroying gpu')
        this.gpu.cleanup()
    }


    clipToRay(x,y) {
        const cam = this.camera
        let clip = v4(2*x/this.width - 1, 1 - 2*y/this.height,-1,1)
        let eye = cam.projection.inverse().transform(clip)
        let ray = cam.modelview.inverse().transform(v4(eye.x,eye.y,-1,0))
        return v3(ray.x,ray.y,ray.z).normalized()
    }
    
    async grabParticle(x, y) {
        const { gpu, camera } = this
        let ray = this.clipToRay(x,y)
        let rsq = (phys.r)**2
        let particles = new Particles(await gpu.read(this.bufs.particles))
        
        let hitdists = []
        for (const [i, p] of enumerate(particles)) {
            if (this.meshes[p.mesh].inactive == 1) continue
            let co = camera.pos.sub(p.pos)
            let b = ray.dot(co)
            let discrim = b*b - co.dot(co) + rsq
            if (discrim < 0) continue
            let dist = -b - sqrt(discrim)
            if (dist > 0) hitdists.push([i,dist])
        }
        camera.selection = -1
        this.grab = undefined
        if (hitdists.length == 0) return
        hitdists.sort((a,b) => a[1]-b[1])
        camera.selection = hitdists[0][0]
        let axes = this.camAxes()
        let p = particles[camera.selection]
        this.grab = { p, depth:p.pos.sub(camera.pos).dot(axes.fwd), w:p.w, last:0 }
        p.w = 0
    }

    moveParticle(x, y, drop) {
        if (this.grab == undefined) return
        let { camera, gpu, bufs } = this
        let r = this.clipToRay(x,y)
        let axes = this.camAxes()
        let c = this.grab.depth/r.dot(axes.fwd)
        let R = r.mulc(c)
        let pos = camera.pos.add(R)
        let p = this.grab.p
        
        if (drop) {
            p.w = this.grab.w
            this.grab = undefined
        } else {
            p.vel = pos.sub(p.pos).divc(max(0.01,clock() - this.grab.last))
            p.w = 0
            this.grab.last = clock()
        }
        p.pos = pos
        gpu.write(gpu.chop(gpu.offset(bufs.particles, Particles.stride * camera.selection), Particle.size), p)
    }

    async fixParticle() {
        const { gpu, camera, bufs } = this
        let pid = camera.selection
        if (pid < 0) return
        let buf = gpu.chop(gpu.offset(bufs.particles, Particles.stride * pid), Particle.size)
        let p = new Particle(await this.gpu.read(buf))
        let wPrev = this.fixed[pid]
        if (wPrev == undefined) {
            this.fixed[pid] = p.w
            p.w = 0
        } else {
            p.w = wPrev
            this.fixed[pid] = undefined
        }
        this.gpu.write(buf, p)
    }


    activateHand(yesno) {
        const { gpu, bufs, params, camera } = this
        this.handPos = camera.pos.sub(v3(0,0,.5)).add(camera.forward.mulc(0.3))
        this.handUD = 0
        const handidx = this.meshes.length - 1
        const buf = gpu.chop(gpu.offset(bufs.meshes, Meshes.stride * handidx), Mesh.size)
        this.meshes[handidx].flags = yesno ? 2 : 1
        gpu.write(buf, this.meshes[handidx])        
    }

    moveHand(x, y, pitch) {
        let dy = -0.005 * y, dx = 0.005 * x
        const handDir = sim.handFwd()
        sim.handPos.x += dy * handDir.x + dx * cos(sim.camLR)
        sim.handPos.y += dy * handDir.y - dx * sin(sim.camLR)
        sim.handPos.z += dy * handDir.z
        sim.handUD += pitch
    }

    rotateCam(dx, dy) {
        sim.camLR += dx
        sim.camUD = clamp(sim.camUD + dy, -PI / 2, PI / 2)
    }

    strafeCam(dx, dy) {
        const delta = v3(-dx * cos(sim.camLR), dx * sin(sim.camLR), -dy)
        sim.camPos = sim.camPos.add(delta)
    }

    advanceCam(delta) {
        const camDir = sim.camFwd()
        sim.camPos.x += delta * camDir.x
        sim.camPos.y += delta * camDir.y
        sim.camPos.z += delta * camDir.z
    }
    
}


export class Compute {
    constructor(sim) {
        this.sim = sim
    }

    async setup() {
        const { gpu, verts, particles, edges, tets, meshes, bufs, tetGroups } = this.sim
        if (particles.length == 0) return

        const pd = ceil(particles.length/gpu.threads)
        const ed = ceil(edges.length/gpu.threads)
        const tetd = ceil(tets.length/gpu.threads)

        const threads = gpu.threads
        const wgsl = (await fetchtext('./compute.wgsl')).interp({ threads, MAXNN })
        
        const shader = await gpu.shader({
            compute:true, wgsl:wgsl, defs:[Particle, Mesh, Params],
            storage:{
                particles:Particles, meshes:Meshes, sorted:u32array, centroidwork:v3array, edges:Edges, debug:f32array, tetgroup:u32array,
                cnts:i32array, cnts_atomic:iatomicarray, work:i32array, shapework:m3Array, tets:Tets
            },
            uniform:{ params:Params }
        })

        const cmds = [gpu.timestamp('')]

        const predict = gpu.computePipe({ shader, entryPoint: 'predict', binds: ['particles', 'meshes', 'params'] })
        cmds.push(
            gpu.computePass({ pipe:predict, dispatch:pd, binds:{ particles:bufs.particles, meshes:bufs.meshes, params:bufs.params } }),
            gpu.timestamp('predict')
        )



        const centroid_prep = gpu.computePipe({ shader, entryPoint: 'centroid_prep', binds: ['meshes', 'particles', 'centroidwork'] })
        const get_centroid = gpu.computePipe({ shader, entryPoint: 'get_centroid', binds: ['meshes', 'centroidwork'] })
        const rotate_prep = gpu.computePipe({ shader, entryPoint: 'rotate_prep', binds: ['meshes', 'particles', 'shapework'] })
        const get_rotate = gpu.computePipe({ shader, entryPoint: 'get_rotate', binds: ['meshes', 'shapework'] })
        for (const [i, m] of enumerate(meshes)) {
            let n = m.pf - m.pi
            if (n <= 0) continue
            if (m.flags == 1) continue
            const centroidWork = gpu.buf({ label: `centroidwork${i}`, type: v3array, size: v3array.stride * n, usage: 'STORAGE' })
            const shapeWork = gpu.buf({ label: `shapework${i}`, type: m3Array, size: m3Array.stride * n, usage: 'STORAGE' })
            let dp1 = ceil(n / threads), dp2 = ceil(dp1 / threads)
            const meshBind = { meshes: gpu.offset(bufs.meshes, Meshes.stride * i) }
            cmds.push(gpu.computePass({ pipe: centroid_prep, dispatch: dp1, binds: { ...meshBind, particles: bufs.particles, centroidwork: centroidWork } }))
            cmds.push(gpu.computePass({ pipe: get_centroid, dispatch: dp1, binds: { ...meshBind, centroidwork: centroidWork } }))
            if (dp1 > 1) {
                cmds.push(gpu.computePass({ pipe: get_centroid, dispatch: dp2, binds: { ...meshBind, centroidwork: centroidWork } }))
                if (dp2 > 1)
                    cmds.push(gpu.computePass({ pipe: get_centroid, dispatch: 1, binds: { ...meshBind, centroidwork: centroidWork } }))
            }
            if (n <= 1) continue
            cmds.push(gpu.computePass({ pipe: rotate_prep, dispatch: dp1, binds: { ...meshBind, particles: bufs.particles, shapework: shapeWork } }))
            cmds.push(gpu.computePass({ pipe: get_rotate, dispatch: dp1, binds: { ...meshBind, shapework: shapeWork } }))
            if (dp1 > 1) {
                cmds.push(gpu.computePass({ pipe: get_rotate, dispatch: dp2, binds: { ...meshBind, shapework: shapeWork } }))
                if (dp2 > 1)
                    cmds.push(gpu.computePass({ pipe: get_rotate, dispatch: 1, binds: { ...meshBind, shapework: shapeWork } }))
            }
        }
        cmds.push(gpu.timestamp('get rotation'))


        cmds.push(
            gpu.computePass({ 
                pipe:gpu.computePipe({ shader, entryPoint:'shapematch', binds:['particles', 'meshes', 'params', 'debug'] }),
                dispatch:pd, binds:{ particles:bufs.particles, meshes:bufs.meshes, params:bufs.params, debug:bufs.debug },
            }),
            gpu.timestamp('shape match')
        )

        const neohookean = gpu.computePipe({ shader, entryPoint:'neohookean', binds:['particles', 'meshes', 'tets', 'params','debug','tetgroup'] })
        cmds.push(...tetGroups.map(group => gpu.computePass({
            pipe: neohookean, dispatch:ceil(group.data.length/gpu.threads), 
            binds:{ particles:bufs.particles, meshes: bufs.meshes, tets: bufs.tets, params: bufs.params, debug:bufs.debug, tetgroup:group.buf }
        })))
        cmds.push(gpu.timestamp('neohookean'))


        const cntsort_cnt = gpu.computePipe({ shader, entryPoint: 'cntsort_cnt', binds: ['particles', 'cnts_atomic', 'meshes', 'params'] })
        const prefsum_down = gpu.computePipe({ shader, entryPoint: 'prefsum_down', binds: ['cnts', 'work'] })
        const prefsum_up = gpu.computePipe({ shader, entryPoint: 'prefsum_up', binds: ['cnts', 'work'] })
        const cntsort_sort = gpu.computePipe({ shader, entryPoint: 'cntsort_sort', binds: ['particles', 'meshes', 'cnts_atomic', 'sorted'] })
        const grid_collide = gpu.computePipe({ shader, entryPoint: 'grid_collide', binds: ['particles', 'meshes', 'cnts', 'sorted', 'params'] })       
        cmds.push(
            gpu.clearBuffer(bufs.cnts, 0, bufs.cnts.size),
            gpu.computePass({ pipe:cntsort_cnt, dispatch:pd, binds:{ particles:bufs.particles, cnts_atomic:bufs.cnts, meshes:bufs.meshes, params:bufs.params }}),
            gpu.timestamp('cntsort_cnt'),
            gpu.computePass({ pipe:prefsum_down, dispatch:threads ** 2, binds:{ cnts:bufs.cnts, work:bufs.work1 }}),
            gpu.computePass({ pipe:prefsum_down, dispatch:threads, binds:{ cnts:bufs.work1, work:bufs.work2 }}),
            gpu.computePass({ pipe:prefsum_down, dispatch:1, binds:{ cnts:bufs.work2, work:bufs.work2 }}),
            gpu.computePass({ pipe:prefsum_up, dispatch:threads - 1, binds:{ cnts:bufs.work1, work:bufs.work2 }}),
            gpu.computePass({ pipe:prefsum_up, dispatch:threads**2 - 1, binds:{ cnts:bufs.cnts, work:bufs.work1 }}),
            gpu.timestamp('prefsum'),
            gpu.computePass({ pipe:cntsort_sort, dispatch:pd, binds:{ particles:bufs.particles, meshes:bufs.meshes, cnts_atomic:bufs.cnts, sorted:bufs.sorted }}),
            gpu.timestamp('cntsort_sort'),
            gpu.computePass({ pipe:grid_collide, dispatch:pd, binds:{ particles:bufs.particles, meshes:bufs.meshes, cnts:bufs.cnts, sorted:bufs.sorted, params:bufs.params } }),
            gpu.timestamp('find collisions')
        )
        
        const project = gpu.computePipe({ shader, entryPoint: 'project', binds: ['particles', 'meshes', 'params', 'debug'] })
        cmds.push(
            gpu.computePass({ pipe:project, dispatch:pd, binds:{ particles:bufs.particles, meshes:bufs.meshes, params:bufs.params, debug:bufs.debug } }),
            gpu.timestamp('project'),
        )


        cmds.push(
            gpu.computePass({ 
                pipe:gpu.computePipe({ shader, entryPoint:'update_vel', binds:['particles', 'meshes', 'params', 'debug'] }),
                dispatch:pd, binds:{ particles:bufs.particles, meshes:bufs.meshes, params:bufs.params, debug:bufs.debug }
            }),
            gpu.timestamp('update vel')
        )

        this.batch = gpu.encode(cmds)
        this.fwdstep = false
        this.tstart = this.tlast = clock()
        this.frames = this.steps = 0
        this.profiles = []

    }

    async stats() {
        let ret = { kind:'phys', fps:this.frames / (this.tlast - this.tstart) }
        if (this.batch) {
            let data = new BigInt64Array(await this.sim.gpu.read(this.batch.stampBuf))
            let labels = this.batch.stampLabels
            ret.profile = [...range(1,labels.length)].map(i => [labels[i], data[i] - data[i-1]])
        }
        this.tstart = this.tlast
        this.frames = 0
        return ret
    }

    async step(tstep) {
        const { gpu, particles, bufs, handPos, handUD, params, camPos, camLR } = this.sim
        if (particles.length == 0) return
        if (!phys.paused || this.fwdstep) {
            phys.keys.filter(k => phys.type[k] == 'num').forEach(k => { 
                params[k] = phys[k]
            })
            params.t = tstep
            params.step = this.steps
            params.ground = render.ground ? 1 : 0
            params.camPos = camPos
            params.handpos = handPos
            params.handrot = m3(M4.zrot(camLR)).mul(m3(M4.xrot(handUD)))
            gpu.write(bufs.params, params )
            this.batch.execute()
            this.frames++
            this.steps++


            

           
           /*const reads = [
                'particles',
                'cnts'
                'tets',
                'tris',
                'vertices',
                'meshes',
                'debug'
            ].filter(b=>bufs[b])
            
            const data = await Promise.all(reads.map(b => gpu.read(bufs[b])))
            for (let i of range(reads.length))
                globalThis[reads[i]] = new bufs[reads[i]].type(data[i])*/
            
            /*for (let i of range(particles.length)) {
                let p = globalThis.particles[i]
                if (!p.pos.isFinite() || !p.vel.isFinite() || !p.prev_pos.isFinite()) {
                    dbg({msg:'!FINITE', i, p })
                    phys.paused = true
                }
            }*/

            
        }
        this.tlast = clock()            
        this.fwdstep = false        
    }
    
}



export class Render {
    constructor(sim) {
        this.sim = sim
    }
    
    async setup() {
        const { ctx, gpu, bufs, bitmaps, particles, verts, tris, camera, lights, width, height, edges } = this.sim
        gpu.configure(ctx, width, height, render)
        render.watch(render.keys.filter(key => key != 'fov'), () => { this.reset = true })               

        const vd = ceil(verts.length/gpu.threads)
        const td = ceil(tris.length/gpu.threads)

        camera.r = phys.r
        camera.selection = -1
        let wgsl = (await fetchtext('./prerender.wgsl')).interp({threads: gpu.threads}) 
        const preShader = await gpu.shader({ 
            wgsl, compute: true, defs:[ Vertex, Mesh, Particle, Params, TriVert, Triangle ],
            storage:{  particles:Particles, meshes:Meshes, vertices:Vertices, tris:Triangles, tets:Tets },
            uniform: { params:Params } 
        })        

        wgsl = await fetchtext('./render.wgsl')
        wgsl = wgsl.interp({numLights: lights.length })
        const shader = await gpu.shader({ wgsl, defs:[Vertex, Particle, Mesh, Camera, Light],
                                          storage:{ particles:Particles, meshes:Meshes, vertices:Vertices },
                                          uniform:{ camera:Camera, lights:lights.constructor, },
                                          textures:{ tex:{ name:'texture_2d_array<f32>' } },
                                          samplers:{ samp:{ name:'sampler' } } })

        let tex = gpu.texture(bitmaps)
        let samp = gpu.sampler()       

        const cmds = []
        cmds.push(gpu.timestamp(''))
        if (vd > 0 && particles.length > 0) {
            if (bufs.tets.data.length > 0) {
                const vertpos = gpu.computePipe({ shader: preShader, entryPoint:'vertpos', binds: ['vertices','particles','meshes','tets'] })
                cmds.push(gpu.computePass({ pipe:vertpos, dispatch:vd, binds:{ vertices:bufs.vertices, particles:bufs.particles, meshes:bufs.meshes, tets:bufs.tets } }))
                cmds.push(gpu.timestamp('vertexpositions'))
            }
            const normals = gpu.computePipe({ shader: preShader, entryPoint:'normals', binds: ['vertices','particles'] })
            cmds.push(gpu.computePass({ pipe:normals, dispatch:vd, binds:{ vertices:bufs.vertices, particles:bufs.particles } }))
            cmds.push(gpu.timestamp('vertexnormals'))
        }
        if (td > 0) {
            const update_tris = gpu.computePipe({ shader: preShader, entryPoint:'update_tris', binds: ['tris','vertices'] })
            cmds.push(gpu.computePass({ pipe:update_tris, dispatch:td, binds:{ vertices:bufs.vertices, tris:bufs.tris } }))
            cmds.push(gpu.timestamp('vertexnormals'))
        }

        const draws = []
        if (render.ground) {
            const gndPipe = gpu.renderPipe({ shader, entry:'ground', cullMode:'none', binds:['camera','lights'], topology:'triangle-strip' })
            draws.push(gpu.draw({ pipe:gndPipe, dispatch:4, binds:{ camera:bufs.camera, lights:bufs.lights }}))
        }

        if (tris.length > 0) {
            const surfPipe = gpu.renderPipe({ shader, entry:'surface', binds: ['meshes', 'camera', 'lights', 'tex', 'samp'], 
                                              vertBufs: [{ buf:bufs.tris, arrayStride:TriVert.size,
                                                           attributes: [{ shaderLocation:0, offset:TriVert.pos.off, format:'float32x3' },
                                                                        { shaderLocation:1, offset:TriVert.norm.off, format:'float32x3' },
                                                                        { shaderLocation:2, offset:TriVert.mesh.off, format:'uint32' },
                                                                        { shaderLocation:3, offset:TriVert.uv.off, format:'float32x2' }]}]})
            draws.push(gpu.draw({ pipe:surfPipe, dispatch:tris.length*3, binds:{ meshes: bufs.meshes, camera:bufs.camera, lights:bufs.lights, tex, samp }}))
        }
        if (render.particles && particles.length > 0) {
            const partPipe = gpu.renderPipe({ shader, entry:'particle', binds: ['meshes', 'camera', 'lights'], topology: 'triangle-strip',
                                              vertBufs: [{ buf:bufs.particles, arrayStride:Particles.stride, stepMode: 'instance',
                                                           attributes: [{ shaderLocation:0, offset:Particle.pos.off, format:'float32x3' },
                                                                        { shaderLocation:1, offset:Particle.mesh.off, format:'uint32' }]}] })
            draws.push(gpu.draw({ pipe:partPipe, dispatch:[8, particles.length], binds:{ meshes: bufs.meshes, camera:bufs.camera, lights:bufs.lights }}))
        }

        if (render.normals) {
            const normPipe = gpu.renderPipe({ shader, entry:'normals', binds: ['camera'], topology: 'line-list',
                                              vertBufs: [{ buf:bufs.tris, arrayStride:TriVert.size, stepMode: 'instance',
                                                           attributes: [{ shaderLocation:0, offset:TriVert.pos.off, format:'float32x3' },
                                                                        { shaderLocation:1, offset:TriVert.norm.off, format:'float32x3' }]}]})
            draws.push(gpu.draw({ pipe:normPipe, dispatch:[2, tris.length*3], binds:{ camera:bufs.camera }}))
        }

        if (render.edges && edges.length > 0) {
            const edgePipe = gpu.renderPipe({ 
                shader, entry:'edges', binds:['camera'], topology:'line-list',
                indexBuf: { buf: bufs.edges, indexFormat: 'uint32' },
                vertBufs: [{ 
                    buf:bufs.particles, arrayStride:Particles.stride, stepMode:'vertex',
                    attributes: [{ shaderLocation:0, offset:Particle.pos.off, format:'float32x3'}]
                }]
            })
            draws.push(gpu.drawIndexed({ pipe:edgePipe, dispatch:edges.length, binds:{ camera:bufs.camera }}))
        }

        const lightPipe = gpu.renderPipe({ shader, entry:'lights', binds: ['camera','lights'], topology: 'triangle-strip',
                                           cullMode: 'front', atc:false })
        draws.push(gpu.draw({ pipe:lightPipe, dispatch:[14, lights.length], binds: { camera:bufs.camera, lights:bufs.lights }}))

        cmds.push(gpu.renderPass(draws))
        cmds.push(gpu.timestamp('draws'))

        this.batch = gpu.encode(cmds)
        this.profiles = []
        this.tstart = this.tlast = clock()
        this.frames = 0
    }

    async stats() {
        const ret = { kind:'render', fps: this.frames/(this.tlast - this.tstart) }
        if (this.batch) {
            let data = new BigInt64Array(await this.sim.gpu.read(this.batch.stampBuf))
            let labels = this.batch.stampLabels
            ret.profile = [...range(1,labels.length)].map(i => [labels[i], data[i] - data[i-1]])
        }
        this.tstart = this.tlast
        this.frames = 0
        return ret
    }
    
    async step() {
        const { camera, lights, ctx, width, height, gpu, bufs, camPos, params } = this.sim
        if (this.reset) {
            await gpu.dev.queue.onSubmittedWorkDone()
            await this.setup()
            this.reset = false
        }

        camera.t = performance.now() / 1000
        camera.r = phys.r
        camera.pos = camPos
        camera.ratio = width/height
        camera.projection = M4.perspective(render.fov, camera.ratio, .01, 10000)
        camera.forward = this.sim.camFwd()
        camera.modelview = M4.look(camPos, camera.forward, UP)
        camera.inverse = camera.modelview.mul(camera.projection).inverse()
        gpu.write(bufs.camera, camera)
        
        gpu.write(bufs.lights, lights)

        params.camPos = camPos
        gpu.write(bufs.params, params)
        
        this.batch.execute()
        this.frames++
        this.tlast = clock()

    }

}



export const sampleMesh = async (meshId, D, transaction) => {
    let [mesh,verts,faceData] = await Promise.all([
        transaction.query('meshes', { key:meshId }),
        transaction.query('verts', { index:'meshId', key:meshId }),
        transaction.query('faces', { index:'meshId', key:meshId }),
    ])
    mesh = mesh.get(meshId)
    for (let [id,vert] of verts) vert.pos = v3(...vert.pos).mul(mesh.scale).add(mesh.offset)
    let tree = new BVHTree([...faceData].map(([,face]) => [0,1,2].map(i => verts.get(face['vertId'+i]).pos)))
    let bmin = v3(Infinity), bmax = v3(-Infinity)
    for (const [vertId, vert] of verts) {
        vert.loop = new Map()
        bmin = bmin.min(vert.pos)
        bmax = bmax.max(vert.pos)
    }
    let bounds = bmax.sub(bmin)
    dbg({bounds})
    let dims = [...bounds.divc(D)].map(roundEps).map(ceil).map(d => d + (d%2 == 0 ? 1 : (d > 1 ? 2 : 1)))
    dbg({dims})
    let space = v3(...dims).subc(1).mulc(D).sub(bounds).divc(2)
    let offset = bmin.sub(space)
    dbg({offset})
    let [dimx,dimy,dimz] = dims
    let dimxy = dimx*dimy

    let tetsA = [[6,3,5,0], [4,6,5,0], [2,3,6,0], [1,5,3,0], [6,5,3,7]]
    let tetsB = [[1,4,2,0], [1,2,4,7], [7,2,4,6], [4,1,7,5], [2,7,1,3]]

    let hpmap = new Map(), hvmap = new Map(), particles = [], tets = [], h = 0
    for (let [h,[x,y,z]] of enumerate(range3d(...dims))) {
        let p = v3(D*x,D*y,D*z).add(offset)
        if (tree.signedDist(p) <= D*TETLIM) {   
            hpmap.set(h, p)
            if (mesh.fluid && (x+y+z) % 2 == 0 ) particles.push(p)
        }
    }

    const keepTet = (tet) => {
        for (let p of tet)
            if (tree.signedDist(p) <= 0)
                return true
        for (let i of range(4)) {
            let faceCenter = v3(0)
            for (let j of range(4))
                faceCenter = faceCenter.add(tet[j].mulc(Number(i != j)))
            faceCenter = faceCenter.divc(3)
            if (tree.signedDist(faceCenter) <= 0)
                return true
        }
        let [a,b,c,d] = tet
        for (let [start,end] of [[a,b],[a,c],[a,d],[b,c],[b,d],[c,d]]) {
            let ray = end.sub(start)
            let [rayLen,rayDir] = [ray.mag(), ray.normalized()]
            if (tree.traceRay(start, rayDir).t <= rayLen)
                return true
        }
        return false
    }

    if (!mesh.fluid)
        for (let [xi,yi,zi] of range3d(...dims))
            for (let reltet of (xi+yi+zi) % 2 == 1 ? tetsA : tetsB) {
                let xyzs = reltet.map(vid => [xi + (vid&1), yi + Number(Boolean(vid&2)), zi + Number(Boolean(vid&4))])
                if (xyzs.some(([x,y,z]) => x >= dimx || y >= dimy || z >= dimz)) continue
                let hs = xyzs.map(([x,y,z]) => x + y*dimx + z*dimxy)
                let hps = hs.map(h => [h, hpmap.get(h)])
                if (hps.some(([h,p]) => p == undefined)) continue
                if (!keepTet(hps.map(([h,p]) => p))) continue
                tets.push(hps.map(([h,p]) => {
                    let pidx = hvmap.get(h)
                    if (pidx == undefined) {
                        pidx = particles.length
                        particles.push(p)
                        hvmap.set(h, pidx)
                    }
                    return pidx
                }))
            }

    dbg({particles:particles.length})
    dbg({tets:tets.length})

    let cache = { 
        tets: Tets.alloc(tets.length), 
        particles: v3array.alloc(particles.length),
        mass: new Float32Array(particles.length),
        verts: Vertices.alloc(verts.size),
        faces: Triangles.alloc(faceData.size)
    }

    for (let [pidx,pos] of enumerate(particles))
        cache.particles[pidx] = pos

    let centers = []
    let tmats = []
    for (let [tidx,pids] of enumerate(tets)) {
        cache.tets[tidx] = Tet.of(pids)
        let ps = pids.map(pid => particles[pid])
        centers.push(ps[0].add(ps[1]).add(ps[2]).add(ps[3]).divc(4))
        let tmat =  m3([ps[0].sub(ps[3]), ps[1].sub(ps[3]), ps[2].sub(ps[3])])
        let vol = -tmat.invert() / 6
        if (vol < 0) throw new Error('got zero or negative volume')
        let pm = vol / 4
        for (let pid of pids)
            cache.mass[pid] += pm                
        tmats.push(tmat)
    }

    for (let [id,face] of faceData)
        for (let [i,j,k] of [[0,1,2],[1,2,0],[2,0,1]]) 
            verts.get(face['vertId'+i]).loop.set(face['vertId'+j],face['vertId'+k])

    let vertIdMap = new Map()
    for (let [vidx,[vid,vert]] of enumerate(verts))
        vertIdMap[vid] = vidx

    for (let [vidx,[vid,vert]] of enumerate(verts)) {
        let v = cache.verts[vidx]
        v.pos = vert.pos
        let [heads,tails] = [[...vert.loop.keys()], [...vert.loop.values()]]
        heads = heads.filter(head => !tails.includes(head))
        if (heads.length == 0) heads = [tails[0]]
        for (let estart of heads) {
            let evert = estart
            do {
                v.edges[v.nedges++] = vertIdMap[evert]
                evert = vert.loop.get(evert)
            } while (evert != estart && evert != undefined)
        }
        let tetbest = { dist: Infinity }
        for (let t of range(tmats.length)) {
            let dist = centers[t].dist(v.pos)
            if (dist < tetbest.dist) tetbest = { t, dist }
        }
        if (tetbest.dist != Infinity) {
            v.tet = tetbest.t
            v.bary = v.pos.sub(particles[tets[v.tet][3]]).mulm(tmats[tetbest.t])
        } else v.tet = -1
    }

    for (let [fidx,face] of enumerate(faceData.values()))
        cache.faces[fidx] = Triangle.of(...[0,1,2].map(i => TriVert.of(
            v3(0), vertIdMap[face['vertId'+i]], v3(0), -1, v2(...face.uv[i]))))

    let edges = new Set()
    for (let [a,b,c,d] of cache.tets)
        for (let [pid1,pid2] of [[a,b],[a,c],[a,d],[b,c],[b,d],[c,d]])
            edges.add(hashPair(pid1,pid2))
    edges = [...edges].map(hash => unhashPair(hash)).flat()
    cache.edges = u32array.of(edges)

    for (let prop in cache) cache[prop] = cache[prop].buffer
    cache.meshId = meshId

    let tetGroups = []
    let yetGroups = new Set([...range(tets.length)])
    while (yetGroups.size > 0) {
        let group = []
        let groupPids = new Set()
        for (const tid of yetGroups) {
            let pids = [...tets[tid]]
            if (group.length == 0 || !pids.some(pid => groupPids.has(pid))) {
                group.push(tid)
                yetGroups.delete(tid)
                for (const pid of pids)
                    groupPids.add(pid)
            } 
        }
        tetGroups.push(group)
    }
    cache.tetGroups = tetGroups
    transaction.objectStore('cache').put(cache)
    dbg('sampled')
    return cache
}

export const loadWavefront = async (name, data, transaction) => {
    let meshes = transaction.objectStore('meshes')
    let meshId = await transaction.wait(meshes.add({
        name, bitmapId:-1, color:[1,1,1,1], offset:[0,0,0], rotation:[0,0,0], gravity:1, invmass:1,
        scale:[1,1,1], 'shape stiff':1, 'vol stiff':1, friction:1, 'collision damp':1, 'fluid':0
    }))
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
            let mass = vdata.length > 3 ? vdata[3] : 1.0;
            vertIds[localVerts++] = await transaction.wait(verts.add({ pos, mass, meshId }))
        } else if (key == 'vt') {
            let vtdata = toks.map(parseFloat)
            uvIds[localUVs++] = vtdata.slice(0, 2)
        } else if (key == 'f') {
            if (toks.length == 3) {
                let face = toks.map((tok,i) => [`vertId${i}`, vertIds[parseInt(tok.split('/')[0])]])
                let uv = toks.map(tok => uvIds[parseInt(tok.split('/')[1])] || [0,0])
                await transaction.wait(faces.add({ ...Object.fromEntries(face), uv, meshId }))
            } 
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






