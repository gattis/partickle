import * as util from './utils.js'
import * as gpu from './gpu.js'
import * as geo from './geometry.js'
Object.assign(globalThis, util, gpu, geo)

const MAXNN = 18
const MAXEDGES = 18
let UP = v3(0,0,1)
let CAM_POS = v3(0, -2, .5)
let CAM_UD = 0
const particleColors = [v4(.3,.6,.8,1), v4(.99,.44,.57,1), v4(.9, .48, .48)]
const LIGHTS = [
    { power:1.5, color:v3(1,.85,.6), pos:v3(2,2,2.3) },
    { power:1.5, color:v3(1,.85,.6), pos:v3(2,-2,2.3) },
    { power:1.5, color:v3(1,.85,.6), pos:v3(-2,2,2.3) },
    { power:1.5, color:v3(1,.85,.6), pos:v3(-2,-2,2.3) },
]

export const phys = new Preferences('phys')
phys.addBool('paused', false)
phys.addNum('r', .01, 0, 0.1, 0.001)
phys.addNum('density', 1.0, 0.1, 10, 0.1)
phys.addNum('frameratio', 3, 1, 20, 1)
phys.addNum('speed', 1, 0.05, 5, .05)
phys.addNum('gravity', 9.8, -5, 20, 0.1)
phys.addNum('shape_stiff', 0, 0, 1, 0.005)
phys.addNum('vol_stiff', .5, 0, 1, 0.005)
phys.addNum('damp', 0.5, 0, 1, .01)
phys.addNum('collidamp', .1, 0, 1, .01)


export const render = new Preferences('render')
render.addBool('particles', true)
render.addBool('ground', true)
render.addBool('normals', false)
render.addBool('edges', false)
render.addBool('depth_wr', true)
render.addBool('atc', true)
render.addChoice('depth_cmp', 'less-equal', ['less-equal','less','greater-equal','greater','always','never'])
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
        ['selfcollide', i32],
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
const m3Array = GPU.array({ type:M3 })
const V3Array = GPU.array({ type:V3 })
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

        let ids = { particles:{}, bitmaps:{'-1':-1}, meshes:{}, verts:{} }
        let cnts = { particles:0, verts:0, tets:0 }

        db.transact(['bitmaps','meshes','particles','verts','faces','tets'])
        let [bitmapData, meshData, faceData] = await Promise.all([db.query('bitmaps'), db.query('meshes'), db.query('faces')])
        let meshRel = await Promise.all([...meshData.keys()].map(mid => Promise.all([
            db.query('particles', { index:'meshId', key: mid }),
            db.query('tets', { index:'meshId', key: mid }),
            db.query('verts', { index:'meshId', key: mid }),
        ])))
        db.commit()
        let meshes = Meshes.alloc(meshData.size)
        let particles = Particles.alloc(meshRel.map(([parts,,]) => parts.size).sum())
        let tets = Tets.alloc(meshRel.map(([,tets,]) => tets.size).sum())
        let verts = Vertices.alloc(meshRel.map(([,,verts]) => verts.size).sum())
        let tris = Triangles.alloc(faceData.size)
        

        let bitmaps = []
        for (let [bidx,[bid,bdata]] of enumerate(bitmapData)) {
            ids.bitmaps[bid] = bidx
            bitmaps.push(bdata.data)
        }

        let vertFaces = new Map()
        for (let [fid,fdata] of faceData) {
            let [vid0,vid1,vid2] = [...range(3)].map(i => fdata['vertId'+i])
            vertFaces.setDefault(vid0,[]).push([vid1,vid2])
            vertFaces.setDefault(vid1,[]).push([vid2,vid0])
            vertFaces.setDefault(vid2,[]).push([vid0,vid1])
        }

        let mtis = []
        for (let [midx,[mid,mdata]] of enumerate(meshData)) {
            ids.meshes[mid] = midx
            let mesh = meshes[midx]
            mesh.color = v4(...mdata.color)
            mesh.pcolor = particleColors[midx % particleColors.length]
            mesh.gravity = mdata.gravity
            mesh.tex = ids.bitmaps[mdata.bitmapId]
            mesh.rot = m3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            
            let [partData, tetData, vertData] = meshRel[midx]

            mesh.pi = cnts.particles
            for (let [pid,pdata] of partData) {
                let pidx = cnts.particles++
                ids.particles[pid] = pidx
                const p = particles[pidx]
                p.pos = p.prev_pos = p.rest_pos = v3(...pdata.pos).mul(mdata.scale).add(mdata.offset)
                mesh.c0 = mesh.c0.add(p.pos)
                p.mesh = midx
            }
            mesh.pf = cnts.particles
            mesh.c0 = mesh.ci = mesh.c0.divc(mesh.pf - mesh.pi)
            
            let ti = cnts.tets
            let tmats = [], centers = []
            for (let [tid,tdata] of tetData) {
                let tidx = cnts.tets++
                let pids = [0,1,2,3].map(i => ids.particles[tdata['partId'+i]])
                tets[tidx] = Tet.of(pids)
                let ps = pids.map(pid => particles[pid].pos)
                let m = tetVolume(...ps) / 4
                for (let pid of pids)
                    particles[pid].w += m
                tmats.push(m3([ps[0].sub(ps[3]), ps[1].sub(ps[3]), ps[2].sub(ps[3])]).inverse())
                centers.push(ps[0].add(ps[1]).add(ps[2]).add(ps[3]).divc(4))
            }

            for (let [pid,pdata] of partData) {
                const p = particles[ids.particles[pid]]
                if (p.w != 0) p.w = 1 / p.w
                p.q = p.pos.sub(mesh.c0)
            }

            mesh.vi = cnts.verts
            for (let vid of vertData.keys()) ids.verts[vid] = cnts.verts++
            mesh.vf = cnts.verts

            for (const [vid,vdata] of vertData) {
                const v = verts[ids.verts[vid]]
                v.pos = v3(...vdata.pos).mul(mdata.scale).add(mdata.offset)
                v.mesh = midx
                const loop = new Map()
                for (const [vidb,vidc] of vertFaces.getDefault(vid,[]))
                    loop.set(ids.verts[vidb], ids.verts[vidc])
                let [heads,tails] = [[...loop.keys()], [...loop.values()]]
                heads = heads.filter(head => !tails.includes(head))
                if (heads.length == 0) heads = [tails[0]]
                for (let estart of heads) {
                    let evert = estart
                    do {
                        v.edges[v.nedges++] = evert
                        evert = loop.get(evert)
                    } while (evert != estart && evert != undefined)
                }
                let tetbest = { dist: Infinity }
                for (let t of range(tmats.length)) {
                    let dist = centers[t].dist(v.pos)
                    if (dist < tetbest.dist) tetbest = { t, dist }
                }
                if (tetbest.dist != Infinity) {
                    v.tet = ti + tetbest.t
                    v.bary = v.pos.sub(particles[tets[v.tet][3]].pos).mulm(tmats[tetbest.t])
                } else v.tet = -1
                v.q = v.pos.sub(mesh.c0)
            }

        }

        for (let [fidx,[fid,fdata]] of enumerate(faceData))
            tris[fidx] = Triangle.of(...[0,1,2].map(i => TriVert.of(v3(0), ids.verts[fdata['vertId'+i]], v3(0), ids.meshes[fdata.meshId], v2(...fdata.uv[i]))))


       let edges = [], edgeSet = new Set()
       for (let [a,b,c,d] of tets)
            for (let edge of [[a,b],[a,c],[a,d],[b,c],[b,d],[c,d]]) {
                if (edgeSet.has(hashTuple(edge))) continue
                edges.push(...edge)
                edgeSet.add(hashTuple(edge))
            }
        edges = u32array.of(edges)

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
            gpu.buf({ label:'cnts', type:GPU.array({ type:i32, length:threads**3 }), usage:'STORAGE|COPY_DST' }),
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
                particles:Particles, meshes:Meshes, sorted:u32array, centroidwork:V3Array, edges:Edges, debug:f32array, tetgroup:u32array,
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
            const centroidWork = gpu.buf({ label: `centroidwork${i}`, type: V3Array, size: V3Array.stride * n, usage: 'STORAGE' })
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

        cmds.push(
            gpu.computePass({ 
                pipe:gpu.computePipe({ shader, entryPoint:'update_vel', binds:['particles', 'meshes', 'params', 'debug'] }),
                dispatch:pd, binds:{ particles:bufs.particles, meshes:bufs.meshes, params:bufs.params, debug:bufs.debug }
            }),
            gpu.timestamp('update vel')
        )

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
        
        const project = gpu.computePipe({ shader, entryPoint: 'project', binds: ['particles', 'meshes', 'params'] })
        cmds.push(
            gpu.computePass({ pipe:project, dispatch:pd, binds:{ particles:bufs.particles, meshes:bufs.meshes, params:bufs.params } }),
            gpu.timestamp('project'),
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


            
            /*const reads = ['particles','tets','vertices','meshes','debug'].filter(b=>bufs[b])
            const data = await Promise.all(reads.map(b => gpu.read(bufs[b])))
            for (let i of range(reads.length))
                globalThis[reads[i]] = new bufs[reads[i]].type(data[i])*/
            
            
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














