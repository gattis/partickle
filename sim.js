import * as util from './utils.js'
import * as gpu from './gpu.js'
import * as geo from './geometry.js'
Object.assign(globalThis, util, gpu, geo)

const QUINN = { url:'quinn.obj', texUrl:'quinn.png', offset:v3(0,0,.1), tetUrl:'tetquinn.obj' }
const TORUS = { url:'torus.obj', offset:v3(0,0,0.5), color:v4(.7,.2,.1,.89), tetUrl:'tettorus.obj'  }
const GROUND = { url:'ground.obj', texUrl:'marble.png', gravity:0, offset:v3(0,0,.11), sample:true, density:100 }
const CUBE = { url:'cube.obj', offset:v3(0,0,4), sample:false, dense:true, color:v4(0), shape:0 }
const TRI = { url:'tri.obj', sample:true, color:v4(0.1,0.4,0.6,0.7), gravity:0 }
const WALL = { url:'wall2.obj', gravity:0, sample:true, color:v4(0.2,0.2,0.2,.4), density:100, particleColor:v4(0,0,0,0) }
const KNOT = { url:'knot.obj', color:v4(.6,.3,.3,1), offset:v3(0,0,3), sample:true, scale:v3(1) }
const HELPER = { url:'helper.obj', scale:v3(2,2,2), sample:false, gravity:0 }
const HAND = { url:'hand.obj', texUrl:'hand.png', color:v4(1, .9, .8, 1), sample:true, gravity:0, flags:1, density:1, offset:v3(0,-2,1)}
const DRAGON = { url:'dragon.obj', color:v4(.7,.4,.1,1), tetUrl:'tetdragon.obj' }
const ICOS = { url:'ico.obj', color:v4(.7, .2, .1, .6),  tetUrl:'tetico.obj' }

const MESHES = [
    //{ url:'particle.obj', color:v4(.5, .5, .3, 1), offset:v3(0, 0, 1) }, 
    //{ url:'t.obj', tetUrl:'tett.obj' },
    DRAGON,
]

const MAXNN = 16
const MAXEDGES = 18
let UP = v3(0,0,1)
let CAM_POS = v3(0, -1, .5)
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
phys.addNum('edge_stiff', 0.5, 0, 1, 0.001)
phys.addNum('shape_stiff', 0.5, 0, 1, 0.001)
phys.addNum('tetvol_stiff', 0.5, 0, 1, 0.001)
phys.addNum('damp', 0.5, 0, 1, .01)
phys.addNum('collidamp', .9, 0, 1, .01)


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
        ['shape', f32],
        ['flags', u32],
        ['padding', GPU.array({ type:u32, length:24 })]
    ]
})

export const Vertex = GPU.struct({
    name:'Vertex',
    fields:[
        ['pos', V3],
        ['mesh', u32],
        ['q', V3],
        ['particle', i32],
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
        ['mass', f32],
        ['pos_delta', V3],
        ['ti', i32],
        ['vel', V3],
        ['tf', i32],
        ['prev_vel', V3],
        ['ei', i32],
        ['tmp_vel', V3],
        ['ef',i32],
        ['q', V3],
        ['k', u32],
        ['nn', GPU.array({ type:u32, length:MAXNN })],
        ['dbg',f32],
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

export const TetVerts = GPU.array({ type: u32, length: 4 })

export const Tet = GPU.struct({
    name:'Tet',
    fields:[
        ['verts', TetVerts],
        ['vol0',f32],
    ]
})


const Meshes = GPU.array({ type:Mesh })
const Vertices = GPU.array({ type:Vertex })
const Particles = GPU.array({ type:Particle })
const Triangles = GPU.array({ type:Triangle })
const m3Array = GPU.array({ type:M3 })
const V3Array = GPU.array({ type:V3 })
const Tets = GPU.array({ type:Tet })

export class Sim {

    async init(width, height, ctx) {
        const gpu = new GPU()
        await gpu.init(width,height,ctx,render)

        this.camPos = CAM_POS
        this.camLR = 0
        this.camUD = CAM_UD
        this.handPos = v3(0, 0, 0)
        this.handRot = m3([[1,0,0],[0,1,0],[0,0,1]])
        this.handUD = 0
        
        let meshes = []
        let particles = []
        let verts = []
        let tris = []
        let ptets = []
        let edges = []
        let edges_flat = []
        let tets = []
        const bitmaps = []
        window.tris = tris
        for (const opt of MESHES.concat([HAND])) {
            const data = opt.data || await fetchtext(opt.url)
            const mesh = Mesh.alloc()
            mesh.color = opt.color || v4(1)
            mesh.pcolor = opt.particleColor || particleColors[meshes.length % particleColors.length]
            mesh.flags = opt.flags || 0
            mesh.gravity = 'gravity' in opt ? opt.gravity : 1
            mesh.shape = 'shape' in opt ? opt.shape : 1
            mesh.tex = -1
            mesh.rot = m3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            if (opt.texUrl) {
                const img = new Image()
                img.src = opt.texUrl
                await img.decode()
                const bitmap = await createImageBitmap(img)
                mesh.tex = bitmaps.length
                bitmaps.push(bitmap)
            }

            let parsed = parseObj(data)
            let mverts = parsed.verts, mfaces = parsed.faces
            if ('scale' in opt) mverts = mverts.map(v => v.mul(opt.scale))
            if ('offset' in opt) mverts = mverts.map(v => v.add(opt.offset))
            const vertedges = Array(mverts.length).fill().map(v => [])
            let volume = 0
            for (const tri of mfaces) {
                const [[a],[b],[c]] = tri
                vertedges[a].push([c, b])
                vertedges[b].push([a, c])
                vertedges[c].push([b, a])
                volume += abs(mverts[a].dot(mverts[b].cross(mverts[c])) / 6.0);
                tris.push(Triangle.of(...tri.map(([v, uv]) => TriVert.of(v3(0), v + verts.length, v3(0), 0, uv))))
            }

            let mtets, sample
            if (opt.tetUrl) {
                mtets = parseObj(await fetchtext(opt.tetUrl))
                if ('scale' in opt) mtets.verts = mtets.verts.map(v => v.mul(opt.scale))
                if ('offset' in opt) mtets.verts = mtets.verts.map(v => v.add(opt.offset))
                mtets.ptets = Array(mtets.verts.length).fill().map(v => [])
                mtets.barys = Array(mverts.length)
                mtets.edges = Array(mtets.verts.length).fill().map(v => new Set())
                volume = 0
                let mats = []
                let centers = []
                let tetoffset = tets.length
                for (let tet of mtets.faces) {
                    tet = tet.map(v => v[0])
                    for (const vid of tet) {
                        mtets.ptets[vid].push(tets.length)
                        for (const vid2 of tet)
                            if (vid != vid2) 
                                mtets.edges[vid].add(vid2)
                    }
                    const vs = tet.map(vid => mtets.verts[vid])
                    const vol = tetVolume(...vs)
                    volume += abs(vol)
                    let m = m3([vs[0].sub(vs[3]), vs[1].sub(vs[3]), vs[2].sub(vs[3])]).inverse()
                    mats.push(m)

                    let center = v3(0)
                    for (const v of vs)
                        center = center.add(v)
                    centers.push(center.divc(4))
                    tets.push(Tet.of(TetVerts.of(tet.map(vid => vid + particles.length)), vol))
                }
                for (const [i,v] of enumerate(mverts)) {
                    let best = -1, dbest = Infinity
                    for (const j of range(mtets.faces.length)) {
                        const dist = centers[j].dist(v)
                        if (dist < dbest) {
                            best = j
                            dbest = dist
                        }
                    }
                    mtets.barys[i] = [best+tetoffset, v.sub(mtets.verts[mtets.faces[best][3][0]]).mulm(mats[best])]
                }                
                
            } else if (opt.sample) {
                sample = new VoxelGrid(mverts, mfaces, phys.r*2)
                sample.voxelize()
            }
            
            mesh.vi = verts.length
            for (const i of range(mverts.length)) {
                const v = Vertex.alloc()
                v.pos = mverts[i]
                v.mesh = meshes.length
                v.particle = v.tet = -1
                if (mtets) {
                    if (mtets.barys[i] != undefined) {
                        const [tetid, bary] = mtets.barys[i]
                        v.tet = tetid
                        v.bary = v3(bary[0], bary[1], bary[2])
                    }
                } else v.particle = particles.length + (sample ? sample.vertidxs[i] : i)
                verts.push(v)
                const unsorted = vertedges[i], sorted = []
                if (unsorted.length > MAXEDGES) throw new Error(`meshes must have <= ${MAXEDGES} edges/vertex`)
                if (unsorted.length == 0) continue
                if (unsorted.length == 1) {
                    v.nedges = 2
                    v.edges[0] = unsorted[0][0] + mesh.vi
                    v.edges[1] = unsorted[0][1] + mesh.vi
                    continue
                }
                let first = unsorted.findIndex(e1 => unsorted.findIndex(e2 => e1[0] == e2[1]) == -1)
                let cycle = first == -1
                first = max(0, first)
                let nexti = unsorted[first][0]
                while (unsorted.length > 0) {
                    sorted.push(nexti)
                    const found = unsorted.findIndex(([s, f]) => s == nexti)
                    if (found == -1) break
                    nexti = unsorted[found][1]
                    unsorted.splice(found, 1)
                }
                if (!cycle) sorted.push(nexti)
                v.nedges = sorted.length
                for (let e of range(sorted.length))
                    v.edges[e] = sorted[e] + mesh.vi

            }
            mesh.vf = verts.length

            let c = v3(0)
            const density = 1e6 * ('density' in opt ? opt.density : 1)
            let totmass = volume * density
            const mparts = sample ? sample.samples : (mtets ? mtets.verts : mverts)
            mesh.pi = particles.length
            for (const [i, pos] of enumerate(mparts)) {
                const p = Particle.alloc()
                p.pos = p.prev_pos = p.rest_pos = pos
                c = c.add(pos)
                p.mesh = meshes.length
                p.mass = totmass / mparts.length;
                p.ti = p.tf = -1
                particles.push(p)
                if (mtets) {
                    p.mass = 0
                    const mptets = mtets.ptets[i]
                    p.ti = ptets.length
                    for (let tid of mptets) {
                        ptets.push(tid)
                        p.mass += abs(tets[tid].vol0 / 4) * density
                    }
                    p.tf = ptets.length
                    const mpedges = mtets.edges[i]
                    p.ei = edges.length
                    for (let pid of mpedges) {
                        edges.push(pid + mesh.pi)
                        edges_flat.push(i + mesh.pi, pid + mesh.pi)
                    }
                    p.ef = edges.length
                } else if (sample) {
                    const sedges = sample.edges[i]
                    p.ei = edges.length
                    for (const pid of sedges) {
                        edges.push(pid + mesh.pi)
                        edges_flat.push(i + mesh.pi, pid + mesh.pi)
                    }
                    p.ef = edges.length
                } else {
                    let v = verts[i+mesh.vi]
                    p.nedges = v.nedges
                    for (let e of range(v.nedges))
                        p.edges[e] = verts[v.edges[e]].particle
                }
            }
            mesh.pf = particles.length
            mesh.c0 = mesh.ci = c.divc(mparts.length)
            meshes.push(mesh)
        }

        meshes = Meshes.of(meshes)
        verts = Vertices.of(verts)
        particles = Particles.of(particles)
        tris = Triangles.of(tris)
        tets = Tets.of(tets.length > 0 ? tets : [Tet.alloc()])
        ptets = u32array.of(ptets.length > 0 ? ptets : [0])
        edges = u32array.of(edges.length > 0 ? edges : [0])
        edges_flat = u32array.of(edges_flat.length > 0 ? edges_flat:[0])

        for (const part of particles) 
            part.q = part.pos.sub(meshes[part.mesh].c0)
        for (const vert of verts)
            vert.q = vert.pos.sub(meshes[vert.mesh].c0)

        const params = Params.alloc()

        const camera = Camera.alloc()
        const lights = GPU.array({ type:Light, length:LIGHTS.length }).alloc(LIGHTS.length)
        for (const [i,l] of enumerate(LIGHTS)) {
            lights[i].color = l.color
            lights[i].pos = l.pos
            lights[i].power = l.power
        }
        
        console.log(`nparts=${particles.length} nverts=${verts.length} nmeshes=${meshes.length} ntris=${tris.length}`)
        const threads = gpu.threads
        const pd = ceil(particles.length/gpu.threads)
        const vd = ceil(verts.length/gpu.threads)
        const td = ceil(tris.length/gpu.threads)
        
        const buflist = [
            gpu.buf({ label:'particles', data:particles, usage:'STORAGE|COPY_SRC|COPY_DST|VERTEX' }),
            gpu.buf({ label:'edges', data:edges, usage:'STORAGE|COPY_DST|COPY_SRC' }),
            gpu.buf({ label:'edges_flat', data:edges_flat, usage:'STORAGE|INDEX|COPY_DST|COPY_SRC' }),
            gpu.buf({ label:'tets', data:tets, usage:'STORAGE|COPY_DST|COPY_SRC' }),
            gpu.buf({ label:'ptets', data: ptets, usage: 'STORAGE|COPY_DST|COPY_SRC' }),
            gpu.buf({ label:'vertices', data:verts, usage:'STORAGE|VERTEX|COPY_SRC' }),
            gpu.buf({ label:'meshes', data:meshes, usage:'STORAGE|COPY_DST|COPY_SRC' }),
            gpu.buf({ label:'camera', data:camera, usage:'UNIFORM|COPY_DST' }),
            gpu.buf({ label:'params', data:params, usage:'UNIFORM|COPY_DST' }),
            gpu.buf({ label:'tris', data:tris, usage:'STORAGE|VERTEX|COPY_SRC|COPY_DST' }),
            gpu.buf({ label:'cnts', type:GPU.array({ type:i32, length:threads**3 }), usage:'STORAGE|COPY_DST' }),
            gpu.buf({ label:'work1', type:GPU.array({ type:i32, length:threads**2 }), usage:'STORAGE' }),
            gpu.buf({ label:'work2', type:GPU.array({ type:i32, length:threads }), usage:'STORAGE' }),
            gpu.buf({ label:'sorted', type:GPU.array({ type:u32, length:particles.length}), usage:'STORAGE' }),
            gpu.buf({ label:'lights', data:lights, usage:'UNIFORM|FRAGMENT|COPY_DST' })
        ]
        const bufs = Object.fromEntries(buflist.map(buf => [buf.label,buf]))
        Object.assign(this, {gpu, meshes, verts, particles, tris, bitmaps, camera, lights, bufs,
                             ctx, width, height, params, pd, vd, td, edges, edges_flat, fixed:{} })

        this.compute = new Compute(this)
        this.render = new Render(this)
        
        await this.compute.setup()
        await this.render.setup()

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
        const { gpu, compute, render } = this
        while (true) {
            const p = new Promise(resolve => requestAnimationFrame(resolve))
            const tstart = clock()
            await render.step()
            for (let i of range(phys.frameratio))
                await compute.step(1/60/phys.frameratio * phys.speed)
            await p
        }
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
        let rsq = (phys.r*2)**2
        let particles = new Particles(await gpu.read(this.bufs.particles))
        
        let hitdists = []
        for (const [i, p] of enumerate(particles)) {
            if (this.meshes[p.mesh].flags != 0) continue
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
        this.grab = { p, depth:p.pos.sub(camera.pos).dot(axes.fwd), mass:p.mass, last:0 }
        p.mass = 0
        console.log('selected', camera.selection)
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
            p.mass = this.grab.mass
            this.grab = undefined
        } else {
            p.vel = pos.sub(p.pos).divc(max(0.01,clock() - this.grab.last))
            p.mass = 0
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
        let mPrev = this.fixed[pid]
        if (mPrev == undefined) {
            this.fixed[pid] = p.mass
            p.mass = 0
        } else {
            p.mass = mPrev
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


export const parseObj = (data) => {
    const sections = { v:[], vt:[], f:[] }
    data.split(/[\r\n]/).forEach(line => {
        const [key, ...toks] = line.split(/\s/)
        if (key in sections)
            sections[key].push(toks)
    })
    const verts = sections.v.map(toks => v3(...toks.map(parseFloat)))
    const tex = sections.vt.map(toks => v2(...toks.map(parseFloat)))
    const faces = sections.f.map(toks => toks.map(tok => {
        let [v, vt] = tok.split('/').slice(0, 2).map(idx => parseInt(idx) - 1)
        return [v, isNaN(vt) ? v2(0) : tex[vt]]
    }))
    return { verts, faces }
}


export class Compute {
    constructor(sim) {
        this.sim = sim
    }

    async setup() {
        const { gpu, verts, particles, meshes, bufs, pd } = this.sim
        if (particles.length == 0) return
        const threads = gpu.threads
        const wgsl = (await fetchtext('./compute.wgsl')).interp({ threads, MAXNN })
        
        const shader = await gpu.shader({
            compute:true, wgsl:wgsl, defs:[Particle, Mesh, Tet, Params],
            storage:{
                particles:Particles, meshes:Meshes, sorted:u32array, centroidwork:V3Array, ptets:u32array, edges:u32array,
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


        const constrain = gpu.computePipe({ shader, entryPoint: 'constrain', binds: ['particles', 'meshes', 'tets', 'ptets', 'edges', 'params'] })
        const syncpos = gpu.computePipe({ shader, entryPoint: 'syncpos', binds: ['particles', 'meshes', 'params'] })
        cmds.push(
            gpu.computePass({ pipe: constrain, dispatch: pd, binds: { particles:bufs.particles, meshes: bufs.meshes, tets: bufs.tets, ptets: bufs.ptets, edges:bufs.edges, params: bufs.params } }),
            gpu.computePass({ pipe: syncpos, dispatch: pd, binds: { particles: bufs.particles, meshes: bufs.meshes, params: bufs.params } }),
            gpu.timestamp('constrain')
        )

        

        
        this.batch = gpu.encode(cmds)
        this.fwdstep = false
        this.tstart = this.tlast = clock()
        this.frames = 0
        this.profiles = []

    }

    async stats() {
        let ret = { kind:'phys', fps:this.frames / (this.tlast - this.tstart) }
        if (this.batch) {
            let data = new BigInt64Array(await this.sim.gpu.read(this.batch.stampBuf))
            let labels = this.batch.stampLabels
            ret.profile = Array.from(range(1,labels.length)).map(i => [labels[i], data[i] - data[i-1]])
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
            params.ground = render.ground ? 1 : 0
            params.camPos = camPos
            params.handpos = handPos
            params.handrot = m3(M4.zrot(camLR)).mul(m3(M4.xrot(handUD)))
            gpu.write(bufs.params, params )
            this.batch.execute()
            this.frames++
            
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
        const { ctx, gpu, bufs, bitmaps, particles, edges_flat, tris, camera, lights, vd, td, width, height } = this.sim
        gpu.configure(ctx, width, height, render)
        render.watch(render.keys.filter(key => key != 'fov'), () => { this.reset = true })               
        camera.r = phys.r
        camera.selection = -1
        let wgsl = (await fetchtext('./prerender.wgsl')).interp({threads: gpu.threads}) 
        const preShader = await gpu.shader({ 
            wgsl, compute: true, defs:[ Vertex, Mesh, Particle, Params, TriVert, Triangle, Tet ],
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
        if (vd > 0) {
            const vertpos = gpu.computePipe({ shader: preShader, entryPoint:'vertpos', binds: ['vertices','particles','meshes','tets'] })
            const normals = gpu.computePipe({ shader: preShader, entryPoint:'normals', binds: ['vertices','particles'] })
            cmds.push(gpu.computePass({ pipe:vertpos, dispatch:vd, binds:{ vertices:bufs.vertices, particles:bufs.particles, meshes:bufs.meshes, tets:bufs.tets } }))
            cmds.push(gpu.timestamp('vertexpositions'))
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

        if (render.edges) {
            const edgePipe = gpu.renderPipe({ 
                shader, entry:'edges', binds:['camera'], topology:'line-list',
                indexBuf: { buf: bufs.edges_flat, indexFormat: 'uint32' },
                vertBufs: [{ 
                    buf:bufs.particles, arrayStride:Particles.stride, stepMode:'vertex',
                    attributes: [{ shaderLocation:0, offset:Particle.pos.off, format:'float32x3'}]
                }]
            })
            draws.push(gpu.drawIndexed({ pipe:edgePipe, dispatch:edges_flat.length, binds:{ camera:bufs.camera }}))
        }


        const lightPipe = gpu.renderPipe({ shader, entry:'lights', binds: ['camera','lights'], topology: 'triangle-strip',
                                           atc: false, color_src: 'one', color_dst:'one', alpha_src:'one', alpha_dst:'one' })
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
            ret.profile = Array.from(range(1,labels.length)).map(i => [labels[i], data[i] - data[i-1]])
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

        //globalThis.tris = new Triangles(await gpu.read(bufs.tris))
        //globalThis.particles = new Particles(await gpu.read(bufs.particles))
        //globalThis.ptets = new u32array(await gpu.read(bufs.ptets))
        //globalThis.edges = new u32array(await gpu.read(bufs.edges))
        //globalThis.tets = new Tets(await gpu.read(bufs.tets))
        //globalThis.verts = new Vertices(await gpu.read(bufs.vertices))
        //globalThis.meshes = new Meshes(await gpu.read(bufs.meshes))*/

        
            
        
    }

}














