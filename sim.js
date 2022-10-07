import * as util from './utils.js'
import * as gpu from './gpu.js'
import * as geo from './geometry.js'
Object.assign(globalThis, util, gpu, geo)

const MAXNN = 16
const MAXEDGES = 18

let UP = v3(0,0,1)
let CAM_POS = v3(0, -6, 4)
let CAM_UD = -PI/10

export const phys = new Preferences('phys')
phys.addBool('paused', false)
phys.addNum('r', .01, 0, 0.1, 0.001)
phys.addNum('density', 1.0, 0.1, 10, 0.1)
phys.addNum('frameratio', 3, 1, 20, 1)
phys.addNum('speed', 1, 0, 5, .1)
phys.addNum('gravity', 9.8, -5, 20, 0.1)
phys.addNum('spring_stiff', 0.5, 0, 1, 0.01)
phys.addNum('shape_stiff', 0.01, 0, 0.1, 0.001)
phys.addNum('damp', 0.5, 0, 1, .01)
phys.addNum('collidamp', .9, 0, 1, .01)

export const render = new Preferences('render')
render.addBool('particles', true)
render.addBool('ground', true)
render.addBool('normals', false)
render.addBool('grads', false)
render.addBool('axes', false)
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

    
    
const particleColors = [v4(.4,.4,.8,1), v4(.83,.54,.47,1), v4(.2, .48, .48)]

const QUINN = { url:'quinn.obj', texUrl:'quinn.png', offset:v3(0,0,.1), sample:false,
                scale:v3(2, 2, 2), gravity:1, color:v4(.7,.6,.5,1) }



const TORUS = { url:'torus.obj', offset:v3(0,0,2), scale:v3(1),
                color:v4(.7,.2,.1,.89), particleColor:v4(.7,.7,.1,1), sample:true }

const GROUND = { url:'ground.obj', texUrl:'marble.png', gravity:0, offset:v3(0,0,.11),
                 sample:true, particleMass:100 }

const CUBE = { url:'cube.obj', offset:v3(0,0,4), sample:false, dense:true, color:v4(0), shape:0}

const TRI = { url:'tri.obj', sample:true, color:v4(0.1,0.4,0.6,0.7), gravity:0 }

const WALL = { url:'wall2.obj', gravity:0, sample:true, color:v4(0.2,0.2,0.2,.4), particleMass:100, particleColor: v4(0,0,0,0) }


const KNOT = { url:'knot.obj', color:v4(.6,.3,.3,1), offset:v3(0,0,3), sample:true, scale:v3(1)}

const HELPER = { url: 'helper.obj', scale: v3(2,2,2), sample: false, gravity:0 }

const HAND = {
    url: 'hand.obj', texUrl: 'hand.png', color: v4(1,.9,.8,1), sample: true, gravity:0, flags: 1, particleMass: 1
}

const lightUpd = (orig,cur,t,off) => {
    let R = 2.5, r = 0.3, l = 0.6, k = .1
    cur.pos.x = R*((1-k)*cos(t+off) - l*k*cos(t*(1-k)/k+off))
    cur.pos.y = R*((1-k)*sin(t+off) - l*k*sin(t*(1-k)/k+off))
}
const LIGHTS = [
    { power: 1.5, color: v3(1,.85,.6), pos:v3(2,2,2.3) },
    { power: 1.5, color: v3(1,.85,.6), pos:v3(2,-2,2.3) },
    { power: 1.5, color: v3(1,.85,.6), pos:v3(-2,2,2.3) },
    { power: 1.5, color: v3(1,.85,.6), pos:v3(-2,-2,2.3) },
]
      
const MESHES = [
    
    //{ url: 'particle.obj', color: v4(.5, .5, .3, 1), offset: v3(0, 0, 2) }, 
    //{ url: 'particle.obj', color: v4(.5, .3, .5, 1), offset: v3(0, 0, 1) },
    { url: 'torus.obj', color: v4(.5, .3, .5, 1), offset: v3(0, 0, 2), sample:true }
]

const clock = () => phys.speed*performance.now()/1000

export const Mesh = GPU.struct({
    name: 'Mesh',
    fields: [
        ['c0', V3],
        ['ci', V3],
        ['pi', u32],
        ['pf', u32],
        ['rot', M3],
        ['tex', i32],
        ['color', V4],        
        ['pcolor', V4],
        ['gravity', f32],
        ['shape', f32],
        ['flags', u32],
        ['padding', GPU.array({ type: u32, length: 24 })]        
    ]
})

export const Vertex = GPU.struct({
    name: 'Vertex',
    fields: [
        ['pos', V3],
        ['mesh', u32],
        ['q', V3],
        ['particle', i32],
        ['norm', V3],
        ['nedges', u32],
        ['edges', GPU.array({ type: u32, length: MAXEDGES })]        
    ]
})
        

export const Particle = GPU.struct({
    name: 'Particle',
    fields: [
        ['sp', V3],
        ['hash', i32],
        ['si', V3],
        ['mesh', u32],
        ['q', V3],
        ['mass',f32],
        ['s0', V3],
        ['nedges', u32],
        ['grad0', V3],
        ['k', u32],
        ['grad', V3],
        ['v', V3],
        ['edges', GPU.array({ type:u32, length:8})],
        ['nn', GPU.array({ type:u32, length:MAXNN })],
    ]
})


export const Camera = GPU.struct({
    name: 'Camera',
    fields: [
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
    name: 'Light',
    fields: [
        ['pos', V3],
        ['dir', V3],
        ['power', f32],
        ['color', V3]
    ]
})

export const Params = GPU.struct({
    name: 'Params',
    fields: [
        ...phys.keys.filter(k => phys.type[k] == 'num').map(k => [k, f32]),
        ['handpos', V3],
        ['handrot', M3],
        ['ground', u32],
        ['grabbing', i32],
        ['t',f32],
    ]
})

export const TriVert = GPU.struct({
    name: 'TriVert',
    fields: [
        ['pos', V3],
        ['vidx', u32],
        ['norm', V3],
        ['mesh', u32],
        ['uv', V2],
        ['dist',f32]
    ]
})
export const Triangle = GPU.struct({
    name: 'Triangle',
    fields: [
        ['v0',TriVert],
        ['v1',TriVert],
        ['v2',TriVert],
    ]
})

export const Meshes = GPU.array({ type: Mesh })
export const Vertices = GPU.array({ type: Vertex })
export const Particles = GPU.array({ type: Particle })
export const Triangles = GPU.array({ type: Triangle })
export const m3Array = GPU.array({ type: M3 })
export const V3Array = GPU.array({ type: V3 })


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
        
        const objs = []
        for (const opt of MESHES.concat([HAND])) {
            const obj = await this.loadObj(opt)
            obj.particleColor ||= particleColors[objs.length % 3]
            if (obj.sample) {
                const grid = new VoxelGrid(obj.verts, obj.faces, phys.r*2)
                grid.voxelize()
                obj.virts = grid.samples
                obj.virtgrads = grid.gradients
                obj.vertToVirt = grid.vertidxs
                obj.partEdges = grid.edges
            }
            objs.push(obj)
        }
        const nverts = objs.sum(obj => obj.verts.length)
        const nparticles = objs.sum(obj => obj.sample ? obj.virts.length : obj.verts.length)
        const ntris = objs.sum(obj => obj.faces.length)
        const meshes = Meshes.alloc(objs.length)
        const verts = Vertices.alloc(nverts)
        const particles = Particles.alloc(nparticles)
        const ntrisPow2 = roundUpPow(ntris,2)
        const tris = Triangles.alloc(ntrisPow2)
        const bitmaps = []
        
        const vertedges = Array(nverts).fill().map(v => [])
        const cnts = { verts: 0, tris: 0, particles: 0, meshes: 0 }
        for (const obj of objs) {
            const meshidx = cnts.meshes++
            const mesh = meshes[meshidx]
            const vertoff = cnts.verts
            mesh.pi = cnts.particles
            mesh.color = obj.color
            mesh.pcolor = obj.particleColor
            mesh.gravity = obj.gravity
            mesh.flags = obj.flags
            mesh.tex = -1
            mesh.rot = m3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            mesh.shape = obj.shape
            if (obj.bitmap) {
                mesh.tex = bitmaps.length
                bitmaps.push(obj.bitmap)
            }

            for (const i of range(obj.verts.length)) {
                const v = verts[cnts.verts++]
                v.pos = obj.verts[i]
                v.mesh = meshidx
                v.particle = cnts.particles + (obj.sample ? obj.vertToVirt[i] : i)
            }

            const objParticles = obj.sample ? obj.virts : obj.verts
            for (const [i,pos] of enumerate(objParticles)) {
                const p = particles[cnts.particles++]
                p.si = p.s0 = pos
                p.mesh = meshidx
                p.mass = obj.particleMass
                if (obj.sample) {
                    p.grad0 = p.grad = obj.virtgrads[i]
                    const edges = obj.partEdges[i]
                    for (const j of range(edges.length))
                        p.edges[j] = edges[j] + mesh.pi
                    p.nedges = edges.length
                }
                
                
            }
            mesh.pf = cnts.particles
            for (const tri of obj.faces) {
                for (const i of range(3)) tri[i].vidx += vertoff
                tris[cnts.tris++] = tri
                const [a,b,c] = [tri.v0.vidx, tri.v1.vidx, tri.v2.vidx]
                vertedges[a].push([c,b])
                vertedges[b].push([a,c])
                vertedges[c].push([b,a])
            }
        }       


        for (const v of range(verts.length)) {
            const unsorted = vertedges[v], sorted = []
            if (unsorted.length > MAXEDGES) throw new Error(`meshes must have <= ${MAXEDGES} edges/vertex`)
            if (unsorted.length == 0) continue
            if (unsorted.length == 1) {
                verts[v].nedges = 2
                verts[v].edges[0] = unsorted[0][0]
                verts[v].edges[1] = unsorted[0][1]
                continue
            }
            let first = unsorted.findIndex(e1 => unsorted.findIndex(e2 => e1[0]==e2[1]) == -1)
            let cycle = false
            if (first == -1) {
                cycle = true
                first = 0
            }               
            let nexti = unsorted[first][0]
            while (unsorted.length > 0) {
                sorted.push(nexti)
                const found = unsorted.findIndex(([i,f]) => i == nexti)
                if (found == -1) break
                nexti = unsorted[found][1]
                unsorted.splice(found, 1)
            }
            if (!cycle) sorted.push(nexti)
            const vert = verts[v]
            vert.nedges = sorted.length
            for (let e of range(sorted.length))
                vert.edges[e] = sorted[e]
            if (!objs[vert.mesh].sample) {
                const p = particles[vert.particle]
                p.nedges = sorted.length
                for (let e of range(sorted.length))
                    p.edges[e] = verts[sorted[e]].particle
            }
            
        }

        for (const mesh of meshes) {
            let c = v3(0)
            for (let p = mesh.pi; p < mesh.pf; p++)
                c = c.add(particles[p].si)
            mesh.c0 = mesh.ci = c.divc(mesh.pf - mesh.pi)
        }
        for (const part of particles) 
            part.q = part.si.sub(meshes[part.mesh].c0)
        for (const vert of verts)
            vert.q = vert.pos.sub(meshes[vert.mesh].c0)
            

        const params = Params.alloc()
        params.ground = render.ground ? 1 : 0
        params.grabbing = -1
        

        const camera = Camera.alloc()
        const lights = GPU.array({ type: Light, length: LIGHTS.length }).alloc(LIGHTS.length)
        for (const [i,l] of enumerate(LIGHTS)) {
            lights[i].color = l.color
            lights[i].pos = l.pos
            lights[i].power = l.power
        }
        

        const threads = gpu.threads
        const pd = ceil(particles.length/gpu.threads)
        const vd = ceil(verts.length/gpu.threads)
        const td = ceil(tris.length/gpu.threads)
        
        const buflist = [
            gpu.buf({ label:'particles', data:particles, usage: 'STORAGE|COPY_SRC|COPY_DST|VERTEX' }),
            gpu.buf({ label:'vertices', data:verts, usage: 'STORAGE|VERTEX|COPY_SRC' }),
            gpu.buf({ label:'meshes', data:meshes, usage: 'STORAGE|COPY_DST|COPY_SRC' }),
            gpu.buf({ label:'camera', data:camera, usage: 'UNIFORM|COPY_DST' }),
            gpu.buf({ label:'params', data:params, usage: 'UNIFORM|COPY_DST' }),
            gpu.buf({ label:'tris', data:tris, usage: 'STORAGE|VERTEX|COPY_SRC|COPY_DST' }),
            gpu.buf({ label:'cnts', type:GPU.array({ type:i32, length: threads**3 }), usage: 'STORAGE|COPY_DST' }),
            gpu.buf({ label:'work1', type:GPU.array({ type:i32, length: threads**2 }), usage: 'STORAGE' }),
            gpu.buf({ label:'work2', type:GPU.array({ type:i32, length: threads }), usage: 'STORAGE' }),
            gpu.buf({ label:'sorted', type:GPU.array({ type:u32, length: particles.length}), usage: 'STORAGE' }),
            gpu.buf({ label:'lights', data: lights, usage: 'UNIFORM|FRAGMENT|COPY_DST' })
        ]
        const bufs = Object.fromEntries(buflist.map(buf => [buf.label,buf]))
        Object.assign(this, {gpu, objs, meshes, verts, particles, tris, bitmaps, camera, lights, bufs,
                             ctx, width, height, params, pd, vd, td, ntris})

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
        let fwd = this.camFwd()
        let right = fwd.cross(v3(0,0,1))
        let up = right.cross(fwd)
        return { fwd, up, right }
    }

    handFwd() {
        return v3(sin(this.camLR) * cos(this.handUD), cos(this.camLR) * cos(this.handUD), sin(this.handUD)).normalized()
    }
    
    async loadObj(opt) {
        const obj = {
            offset:v3(0), scale:v3(1),            
            color:v4(1), flags:0,
            sample:false, gravity:1, particleMass:1, shape:1
        }
        Object.assign(obj, opt)
        obj.virts = []
        const data = obj.data || await (await fetch(obj.url)).text()
        const sections = {v: [], vt: [], f:[]}
        data.split(/[\r\n]/).forEach(line => {
            const [key,...toks] = line.split(/\s/)
            if (key in sections)
                sections[key].push(toks)
        })
        obj.verts = sections.v.map(toks => v3(...toks.map(parseFloat)).mul(obj.scale).add(obj.offset))
        const tex = sections.vt.map(toks => v2(...toks.map(parseFloat)))
        obj.faces = sections.f.map(toks => Triangle.of(...toks.map(tok => {
            const [v,vt] = tok.split('/').slice(0,2).map(idx => parseInt(idx) - 1)
            return TriVert.of(v3(0), v, v3(0), 0, isNaN(vt) ? v2(0) : tex[vt], 0)
        })))
        if (opt.texUrl) {
            const img = new Image()
            img.src = opt.texUrl
            await img.decode()
            obj.bitmap = await createImageBitmap(img)
        }
        return obj
    }

    async run() {
        const { gpu, compute, render } = this
        while (true) {
            const p = new Promise(resolve => requestAnimationFrame(resolve))
            const tstart = clock()
            await render.step()
            for (let i of range(phys.frameratio))
                await compute.step(1/60/phys.frameratio)
            await p
        }
        requestAnimationFrame(step)
    }

    clipToRay(x,y) {
        const cam = this.camera
        let clip = v4(2*x/this.width - 1, 1 - 2*y/this.height,-1,1)
        let eye = cam.projection.inverse().transform(clip)
        let ray = cam.modelview.inverse().transform(v4(eye.x,eye.y,-1,0))
        return v3(ray.x,ray.y,ray.z).normalized()
    }
    
    async grabParticle(x, y) {
        const { gpu, camera, params } = this
        let ray = this.clipToRay(x,y)
        let rsq = (phys.r*2)**2
        let particles = new Particles(await gpu.read(this.bufs.particles))
        
        let hitdists = []
        for (const [i, p] of enumerate(particles)) {
            if (this.meshes[p.mesh].flags != 0) continue
            let co = camera.pos.sub(p.si)
            let b = ray.dot(co)
            let discrim = b*b - co.dot(co) + rsq
            if (discrim < 0) continue
            let dist = -b - sqrt(discrim)
            if (dist > 0) hitdists.push([i,dist])
        }
        camera.selection = -1
        params.grabbing = -1
        if (hitdists.length == 0) return
        hitdists.sort((a,b) => a[1]-b[1])
        camera.selection = hitdists[0][0]
        params.grabbing = hitdists[0][0]
        let vToPart = ray.mulc(hitdists[0][1])
        this.grabDepth = this.camFwd().dot(vToPart)
    }



    async moveParticle(x, y, updvel) {
        const { camera, gpu, bufs, params } = this
        let buf = gpu.chop(gpu.offset(bufs.particles, Particles.stride * camera.selection), Particle.size)
        let p = new Particle(await this.gpu.read(buf))
        let ray = this.clipToRay(x,y)
        let t = this.grabDepth / ray.dot(this.camFwd())
        let pos = camera.pos.add(ray.mulc(t))
        if (updvel) p.v = pos.sub(p.si).divc(max(0.01,clock() - (this.lastDrag || 0)))
        p.si = pos
        gpu.write(buf, p)
    }

    async dragParticle(x, y) {
        if (this.params.grabbing < 0) return
        
        await this.moveParticle(x, y)
        this.lastDrag = clock()
    }

    async dropParticle(x, y) {
        if (this.params.grabbing < 0) return
        await this.moveParticle(x, y, true)
        this.params.grabbing = -1
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
        const { gpu, verts, particles, meshes, bufs, pd } = this.sim
        if (particles.length == 0) return
        const threads = gpu.threads
        const wgsl = (await fetchtext('./compute.wgsl')).interp({ threads, MAXNN })
        
        const shader = await gpu.shader({
            compute: true, wgsl: wgsl, defs: [Particle, Mesh, Params],
            storage: {
                particles: Particles, meshes: Meshes, sorted: u32array, centroidwork: V3Array,
                cnts: i32array, cnts_atomic: iatomicarray, work: i32array, shapework: m3Array
            },
            uniform: { params: Params }
        })

        const predict = gpu.computePipe({ shader, entryPoint: 'predict', binds: ['particles', 'meshes', 'params'] })

        const centroid_prep = gpu.computePipe({ shader, entryPoint: 'centroid_prep', binds: ['meshes', 'particles', 'centroidwork'] })
        const get_centroid = gpu.computePipe({ shader, entryPoint: 'get_centroid', binds: ['meshes', 'centroidwork'] })
        const rotate_prep = gpu.computePipe({ shader, entryPoint: 'rotate_prep', binds: ['meshes', 'particles', 'shapework'] })
        const get_rotate = gpu.computePipe({ shader, entryPoint: 'get_rotate', binds: ['meshes', 'shapework'] })

        const constrain = gpu.computePipe({ shader, entryPoint: 'constrain', binds: ['particles', 'meshes', 'params'] })

        const cntsort_cnt = gpu.computePipe({ shader, entryPoint: 'cntsort_cnt', binds: ['particles', 'cnts_atomic', 'meshes', 'params'] })
        const prefsum_down = gpu.computePipe({ shader, entryPoint: 'prefsum_down', binds: ['cnts', 'work'] })
        const prefsum_up = gpu.computePipe({ shader, entryPoint: 'prefsum_up', binds: ['cnts', 'work'] })
        const cntsort_sort = gpu.computePipe({ shader, entryPoint: 'cntsort_sort', binds: ['particles', 'meshes', 'cnts_atomic', 'sorted'] })
        const grid_collide = gpu.computePipe({ shader, entryPoint: 'grid_collide', binds: ['particles', 'meshes', 'cnts', 'sorted', 'params'] })       
        const project = gpu.computePipe({ shader, entryPoint: 'project', binds: ['particles', 'meshes', 'params'] })

        console.log(`nparts=${particles.length} nverts=${verts.length} threads=${gpu.threads} pd=${pd}`)
        const cmds = [gpu.timestamp('')]

        cmds.push(
            gpu.computePass({ pipe: predict, dispatch: pd, binds: { particles: bufs.particles, meshes: bufs.meshes, params: bufs.params } }),
            gpu.timestamp('predict')
        )

        for (const [i, m] of enumerate(meshes)) {
            let n = m.pf - m.pi
            if (n <= 0) continue
            if (m.flags == 1) continue
            const centroidWork = gpu.buf({ label: `centroidwork${i}`, type: V3Array, size: V3Array.stride * n, usage: 'STORAGE' })
            const shapeWork = gpu.buf({ label: `shapework${i}`, type: m3Array, size: m3Array.stride * n, usage: 'STORAGE' })
            let dp1 = ceil(n / threads), dp2 = ceil(dp1 / threads)
            const meshBind = { meshes: gpu.offset(bufs.meshes, Meshes.stride * i) }
            cmds.push(gpu.computePass({ pipe: centroid_prep, dispatch: dp1, binds: { ...meshBind, particles:bufs.particles, centroidwork:centroidWork }}))
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
            gpu.computePass({ pipe: constrain, dispatch: pd, binds: { particles: bufs.particles, meshes: bufs.meshes, params: bufs.params } }),
            gpu.timestamp('constrain')
        )
            
        cmds.push(
            gpu.clearBuffer(bufs.cnts, 0, bufs.cnts.size),
            gpu.computePass({ pipe:cntsort_cnt, dispatch:pd, binds:{ particles:bufs.particles, cnts_atomic:bufs.cnts, meshes:bufs.meshes, params:bufs.params }}),
            gpu.timestamp('cntsort_cnt'),
            gpu.computePass({ pipe:prefsum_down, dispatch: threads ** 2, binds:{ cnts:bufs.cnts, work:bufs.work1 }}),
            gpu.computePass({ pipe:prefsum_down, dispatch:threads, binds:{ cnts:bufs.work1, work:bufs.work2 }}),
            gpu.computePass({ pipe:prefsum_down, dispatch:1, binds:{ cnts:bufs.work2, work:bufs.work2 }}),
            gpu.computePass({ pipe:prefsum_up, dispatch:threads - 1, binds:{ cnts:bufs.work1, work: bufs.work2 }}),
            gpu.computePass({ pipe:prefsum_up, dispatch:threads**2 - 1, binds:{ cnts:bufs.cnts, work:bufs.work1 }}),
            gpu.timestamp('prefsum'),
            gpu.computePass({ pipe:cntsort_sort, dispatch:pd, binds:{ particles:bufs.particles, meshes:bufs.meshes, cnts_atomic:bufs.cnts, sorted:bufs.sorted }}),
            gpu.timestamp('cntsort_sort'),
            gpu.computePass({ pipe: grid_collide, dispatch: pd, binds: { particles: bufs.particles, meshes: bufs.meshes, cnts: bufs.cnts, sorted: bufs.sorted, params: bufs.params } }),
            gpu.timestamp('find collisions')
        )
        
        /*cmds.push(
            gpu.computePass({ pipe: grads, dispatch: pd, binds: { particles: bufs.particles, meshes: bufs.meshes } }),
            gpu.timestamp('gradients'),
        )*/

        /*cmds.push(
            gpu.computePass({ pipe:collisions, dispatch:pd, binds:{ particles:bufs.particles, params:bufs.params, meshes:bufs.meshes }}),
            gpu.computePass({ pipe:collisions, dispatch:pd, binds:{ particles:bufs.particles, params:bufs.params, meshes:bufs.meshes }}),
            gpu.computePass({ pipe:collisions, dispatch:pd, binds:{ particles:bufs.particles, params:bufs.params, meshes:bufs.meshes }}),
            gpu.timestamp('stabilize'),
        )*/

        cmds.push(
            gpu.computePass({ pipe: project, dispatch:pd, binds: { particles: bufs.particles, meshes: bufs.meshes, params: bufs.params } }),
            gpu.timestamp('project'),
        )
        
        this.batch = gpu.encode(cmds)
        this.fwdstep = false
        this.tstart = this.tlast = clock()
        this.frames = 0
        this.profiles = []

    }

    async stats() {
        let ret = { kind:'phys', fps: this.frames / (this.tlast - this.tstart) }
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
            params.camPos = camPos
            params.ground = render.ground ? 1 : 0
            params.handpos = handPos
            params.handrot = m3(M4.zrot(camLR)).mul(m3(M4.xrot(handUD)))
            gpu.write(bufs.params, params )
            this.batch.execute()
        }
        this.tlast = clock()
        this.frames++        
        this.fwdstep = false        
    }
    
}



export class Render {
    constructor(sim) {
        this.sim = sim
    }
    
    async setup() {
        const { ctx, gpu, bufs, ntris, bitmaps, particles, camera, lights, vd, td, width, height } = this.sim
        gpu.configure(ctx, width, height, render)
        render.watch(render.keys.filter(key => key != 'fov'), () => { this.reset = true })               
        camera.r = phys.r
        camera.selection = -1
        let wgsl = (await fetchtext('./prerender.wgsl')).interp({threads: gpu.threads}) 
        const preShader = await gpu.shader({ wgsl, compute: true, defs:[ Vertex, Mesh, Particle, Params, TriVert, Triangle ],
                                             storage: { particles:Particles, meshes:Meshes, vertices:Vertices, tris:Triangles },
                                             uniform: { params:Params } })        

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
            const vertpos = gpu.computePipe({ shader: preShader, entryPoint:'vertpos', binds: ['vertices','particles','meshes'] })
            const normals = gpu.computePipe({ shader: preShader, entryPoint:'normals', binds: ['vertices','particles'] })
            cmds.push(gpu.computePass({ pipe:vertpos, dispatch:vd, binds:{ vertices:bufs.vertices, particles:bufs.particles, meshes:bufs.meshes} }))
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

        if (ntris > 0) {
            const surfPipe = gpu.renderPipe({ shader, entry:'surface', binds: ['meshes', 'camera', 'lights', 'tex', 'samp'], 
                                              vertBufs: [{ buf:bufs.tris, arrayStride:TriVert.size,
                                                           attributes: [{ shaderLocation:0, offset:TriVert.pos.off, format:'float32x3' },
                                                                        { shaderLocation:1, offset:TriVert.norm.off, format:'float32x3' },
                                                                        { shaderLocation:2, offset:TriVert.mesh.off, format:'uint32' },
                                                                        { shaderLocation:3, offset:TriVert.uv.off, format:'float32x2' }]}]})
            draws.push(gpu.draw({ pipe:surfPipe, dispatch:ntris*3, binds:{ meshes: bufs.meshes, camera:bufs.camera, lights:bufs.lights, tex, samp }}))
        }
        if (render.particles && particles.length > 0) {
            const partPipe = gpu.renderPipe({ shader, entry:'particle', binds: ['meshes', 'camera', 'lights'], topology: 'triangle-strip',
                                              vertBufs: [{ buf:bufs.particles, arrayStride:Particles.stride, stepMode: 'instance',
                                                           attributes: [{ shaderLocation:0, offset:Particle.si.off, format:'float32x3' },
                                                                        { shaderLocation:1, offset:Particle.mesh.off, format:'uint32' }]}] })
            draws.push(gpu.draw({ pipe:partPipe, dispatch:[8, particles.length], binds:{ meshes: bufs.meshes, camera:bufs.camera, lights:bufs.lights }}))

        }
        if (render.axes) {
            const axesPipe = gpu.renderPipe({ shader, entry:'axes', binds:['camera'], topology:'line-list' })
            draws.push(gpu.draw({ pipe:axisPipe, dispatch:[2, 3], binds:{ camera:bufs.camera }}))
        }
        if (render.normals) {
            const normPipe = gpu.renderPipe({ shader, entry:'normals', binds: ['camera'], topology: 'line-list',
                                              vertBufs: [{ buf:bufs.tris, arrayStride:TriVert.size, stepMode: 'instance',
                                                           attributes: [{ shaderLocation:0, offset:TriVert.pos.off, format:'float32x3' },
                                                                        { shaderLocation:1, offset:TriVert.norm.off, format:'float32x3' }]}]})
            draws.push(gpu.draw({ pipe:normPipe, dispatch:[2, ntris*3], binds:{ camera:bufs.camera }}))

        }
        if (render.grads) {
            const gradPipe = gpu.renderPipe({ shader, entry:'normals', binds: ['camera'], topology: 'line-list',
                                              vertBufs: [{ buf:bufs.particles, arrayStride:Particles.stride, stepMode: 'instance',
                                                           attributes: [{ shaderLocation:0, offset:Particle.si.off, format:'float32x3' },
                                                                        { shaderLocation:1, offset:Particle.grad.off, format:'float32x3' }]}]})
            draws.push(gpu.draw({ pipe:gradPipe, dispatch:[2, particles.length], binds:{ camera:bufs.camera }}))
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
        const { camera, lights, ctx, width, height, gpu, bufs, camPos, ntris, params } = this.sim
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
        //globalThis.verts = new Vertices(await gpu.read(bufs.vertices))
        globalThis.meshes = new Meshes(await gpu.read(bufs.meshes))
        
            
        
    }

}














