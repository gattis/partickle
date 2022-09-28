import * as util from './utils.js'
import * as gpu from './gpu.js'
import * as geo from './geometry.js'
Object.assign(globalThis, util, gpu, geo)

const MAXNN = 24
const MAXEDGES = 18

let UP = v3(0,0,1)
let CAM_POS = v3(0, -3, 2)
let CAM_UD = -PI/8

export const phys = new Preferences('phys')
phys.addBool('paused', false)
phys.addNum('d', .01 , 0, 0.1, 0.01)
phys.addNum('frameratio', 3, 1, 10, 1)
phys.addNum('speed', 1, 0, 5, .01)
phys.addNum('gravity', -9.8, -20, 20, 0)
phys.addNum('collision', .5, 0, 1, .01)
phys.addNum('spring', 0.05, 0, 0.15, 0.001)
phys.addNum('shape', .025, 0, .05, .001)
phys.addNum('friction', 1, 0, 2, .01)

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

const QUINN = { url:'quinn.obj', texUrl:'quinn.png', offset:v3(0,0,0), sample:true,
                scale:v3(2, 2, 2), ffield:v3(0, 0, 0), color:v4(.7,.6,.5,1) }

const HAND = { url:'hand.obj', texUrl:'hand.png', color:v4(1,.9,.8,1), 
               offset:v3(.7,-.5,.6), scale:v3(1,1,1),
               sample:true, ffield:v3(0) }

const TORUS = { url:'torus.obj', offset:v3(0,0,2), scale:v3(1),
                color:v4(.7,.2,.1,.89), particleColor:v4(.7,.7,.1,1), sample:true }

const GROUND = { url:'ground.obj', texUrl:'marble.png', ffield:v3(0), offset:v3(0,0,.11),
                 sample:true, lock:()=>0.99 }

const CUBE = { url:'cube.obj', offset:v3(0,0,4), sample:true, dense:true, color:v4(0), shape:0}

const TRI = { url:'tri.obj', sample:true, color:v4(0.1,0.4,0.6,0.7), ffield:v3(0) }

const WALL = { url:'wall2.obj', ffield:v3(0), sample:true, color:v4(0.2,0.2,0.2,.4), lock:()=>0.99,
             particleColor: v4(0,0,0,0) }


const KNOT = { url:'knot.obj', color:v4(.6,.3,.3,1), offset:v3(0,0,3), sample:true, scale:v3(1)}

const HELPER = { url: 'helper.obj', scale: v3(2,2,2), sample: false, ffield: v3(0) }


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
    //QUINN,HAND,
    KNOT, HAND //GROUND,
    //{ url:objfile, color:v4(1,0,1,1), ffield:v3(0), offset:v3(2.5,5.5,0) },
    //{url:'particle.obj', sample:false, offset:v3(0,0,1)},
    //{url:'particle.obj', sample:false, offset:v3(.1,0,0), ffield: v3(0)},
    
]

const clock = () => phys.speed*performance.now()/1000

export const Mesh = GPU.struct({
    name: 'Mesh',
    fields: [
        ['c0', V3],
        ['tex', i32],
        ['ci', V3],
        ['pi', u32],
        ['pcolor', V4],
        ['pf', u32],
        ['ffield', V3],
        ['shape', f32],
        ['color', V4],        
        ['rot', M3],
        ['padding', GPU.array({ type: u32, length: 28 })]        
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
        ['v', V3],
        ['k', u32],
        ['q', V3],
        ['lock',f32],
        ['s0', V3],
        ['nedges', u32],
        ['grad0', V3],
        ['grad', V3],
        ['edges', GPU.array({ type:u32, length:6})],
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
        ['camPos', V3],
        ['ffield', V3],
        ['collision', f32],
        ['spring', f32],
        ['shape', f32],
        ['friction', f32],
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
        
        const objs = []
        for (const opt of MESHES) {
            const obj = await this.loadObj(opt)
            obj.particleColor ||= particleColors[objs.length % 3]
            if (obj.sample) {
                const grid = new VoxelGrid(obj.verts, obj.faces, phys.d)
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
        const possessed = []

        const vertedges = Array(nverts).fill().map(v => [])
        const cnts = { verts: 0, tris: 0, particles: 0, meshes: 0 }
        for (const obj of objs) {
            const meshidx = cnts.meshes++
            const mesh = meshes[meshidx]
            const vertoff = cnts.verts
            mesh.pi = cnts.particles
            mesh.color = obj.color
            mesh.pcolor = obj.particleColor
            mesh.ffield = obj.ffield
            mesh.tex = -1
            mesh.rot = m3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            mesh.shape = obj.shape
            if (obj.possess)
                possessed.push(mesh)                
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
            for (const [i,pos] of enumerate(obj.sample ? obj.virts : obj.verts)) {
                const p = particles[cnts.particles++]
                p.si = p.s0 = pos
                p.mesh = meshidx
                p.lock = obj.lock(pos)
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
                p = particles[vert.particle]
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
        params.collision = phys.collision
        params.shape = phys.shape
        params.friction = phys.friction
        params.spring = phys.spring
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
                             possessed, ctx, width, height, params, pd, vd, td, ntris})

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
    
    async loadObj(opt) {
        const obj = {
            offset: v3(0), scale: v3(1),            
            color: v4(1), 
            sample: false, ffield: v3(0,0,phys.gravity), lock: ()=>0, shape: 1
        }
        Object.assign(obj, opt)
        obj.virts = []
        const data = await (await fetch(obj.url)).text()
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
        await gpu.dev.queue.onSubmittedWorkDone()       
        let frameratio = 5
        const dbg = _('#dbg')
        const step = async stamp => {
            const thefuture = new Promise(resolve => {
                requestAnimationFrame(nextStamp => {
                    resolve(nextStamp - stamp)
                    step(nextStamp)
                })
            })
            const tstart = clock()
            await render.step()
            for (let i of range(frameratio))
                await compute.step(1/60/frameratio)
            await gpu.dev.queue.onSubmittedWorkDone()
            let twork = clock() - tstart
            let tframe = await thefuture
            let surplus = tframe - 1000*twork
            frameratio =clamp(frameratio + (surplus > 1 ? 1 : -1), 3, phys.frameratio)
            
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
        let rsq = (phys.d)**2
        let particles = new Particles(await gpu.read(this.bufs.particles))
        let hitdists = []
        for (const [i,p] of enumerate(particles)) {
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

    async moveParticle(x, y) {
        const { camera, gpu, bufs, params } = this
        if (params.grabbing < 0) return
        let buf1 = gpu.offset(bufs.particles, Particles.stride*camera.selection)
        let buf2 = gpu.chop(buf1, Particle.size)
        let p = new Particle(await this.gpu.read(buf2))
        let ray = this.clipToRay(x,y)
        let t = this.grabDepth / ray.dot(this.camFwd())
        let pos = camera.pos.add(ray.mulc(t))
        //p.v = pos.sub(p.si).divc(params.t)
        p.si = pos
        gpu.write(buf2, p)
    }

    async dropParticle() {
        this.params.grabbing = -1
    }
    
}


export class Compute {
    constructor(sim) {
        this.sim = sim
    }

    async setup() {
        const { gpu, verts, particles, meshes, params, bufs, tris, pd } = this.sim
        if (particles.length == 0) return
        const threads = gpu.threads
        const wgsl = (await fetchtext('./compute.wgsl')).interp({threads, MAXNN, D:phys.d})
        
        const shader = await gpu.shader({ compute: true, wgsl: wgsl, defs: [Particle, Mesh, Params],
                                          storage: { particles:Particles, meshes:Meshes, sorted:u32array, centroidwork:V3Array,
                                                     cnts:i32array, cnts_atomic:iatomicarray, work:i32array, shapework:m3Array },
                                          uniform: { params:Params } })
        const predict = gpu.computePipe({ shader, entryPoint:'predict', binds:['particles','meshes','params']})
        const cntsort_cnt = gpu.computePipe({ shader, entryPoint:'cntsort_cnt', binds: ['particles','cnts_atomic'] })
        const prefsum_down = gpu.computePipe({ shader, entryPoint:'prefsum_down', binds: ['cnts','work'] })
        const prefsum_up = gpu.computePipe({ shader, entryPoint:'prefsum_up', binds: ['cnts','work'] })
        const cntsort_sort = gpu.computePipe({ shader, entryPoint:'cntsort_sort', binds: ['particles','cnts_atomic','sorted'] })
        const grid_collide = gpu.computePipe({ shader, entryPoint:'grid_collide', binds: ['particles','cnts','sorted'] })
        const centroid_init = gpu.computePipe({ shader, entryPoint:'centroid_init', binds: ['meshes','particles','centroidwork'] })
        const centroid = gpu.computePipe({ shader, entryPoint:'getcentroid', binds: ['meshes','centroidwork'] })
        const shapematch_init = gpu.computePipe({ shader, entryPoint:'shapematch_init', binds: ['meshes','particles','shapework'] })
        const shapematch = gpu.computePipe({ shader, entryPoint:'shapematch', binds: ['meshes','shapework'] })
        const grads = gpu.computePipe({ shader, entryPoint:'grads', binds: ['particles','meshes'] })
        const collisions = gpu.computePipe({ shader, entryPoint:'collisions', binds: ['particles','meshes','params'] })
        const project = gpu.computePipe({ shader, entryPoint:'project', binds: ['particles','meshes','params'] })

        const shapestage = []


        console.log(`nparts=${particles.length} nverts=${verts.length} threads=${gpu.threads} pd=${pd}`)
        const cmds = []
        cmds.push(
            gpu.timestamp(''),
            gpu.clearBuffer(bufs.cnts, 0, bufs.cnts.size),
            gpu.timestamp('clear cnts'),
            gpu.computePass({ pipe:predict, dispatch:pd, binds:{ particles:bufs.particles, meshes:bufs.meshes, params:bufs.params }}),
            gpu.timestamp('predict'),
            gpu.computePass({ pipe:cntsort_cnt, dispatch:pd, binds:{ particles:bufs.particles, cnts_atomic:bufs.cnts }}),
            gpu.timestamp('cntsort_cnt'),
            gpu.computePass({ pipe:prefsum_down, dispatch:threads**2, binds:{ cnts:bufs.cnts, work:bufs.work1 }}),
            gpu.computePass({ pipe:prefsum_down, dispatch:threads, binds:{ cnts:bufs.work1, work:bufs.work2 }}),
            gpu.computePass({ pipe:prefsum_down, dispatch:1, binds:{ cnts:bufs.work2, work:bufs.work2 }}),
            gpu.computePass({ pipe:prefsum_up, dispatch:threads - 1, binds:{ cnts:bufs.work1, work: bufs.work2 }}),
            gpu.computePass({ pipe:prefsum_up, dispatch:threads**2 - 1, binds:{ cnts:bufs.cnts, work:bufs.work1 }}),
            gpu.timestamp('prefsum'),
            gpu.computePass({ pipe:cntsort_sort, dispatch:pd, binds:{ particles:bufs.particles, cnts_atomic:bufs.cnts, sorted:bufs.sorted }}),
            gpu.timestamp('cntsort_sort'),
            gpu.computePass({ pipe:grid_collide, dispatch:pd, binds:{ particles:bufs.particles, cnts:bufs.cnts, sorted:bufs.sorted } }),
            gpu.timestamp('find collisions')
        )
        
        for (const [i,m] of enumerate(meshes)) {
            let n = m.pf - m.pi
            if (n <= 1) continue
            const centroidwork = gpu.buf({ label:`centroidwork${i}`, type:V3Array, size:V3Array.stride*n, usage: 'STORAGE' })
            const shapework = gpu.buf({ label:`shapework${i}`, type:m3Array, size: m3Array.stride*n, usage: 'STORAGE' })
            let dp1 = ceil(n/threads), dp2 = ceil(dp1/threads)
            const meshbind = gpu.offset(bufs.meshes, Meshes.stride*i)
            cmds.push(gpu.computePass({ pipe:centroid_init, dispatch:dp1, binds: { meshes:meshbind, particles:bufs.particles, centroidwork }}))
            cmds.push(gpu.computePass({ pipe:centroid, dispatch:dp1, binds: { meshes:meshbind, centroidwork }}))
            if (dp1 > 1) {
                cmds.push(gpu.computePass({ pipe:centroid, dispatch:dp2, binds: { meshes:meshbind, centroidwork }}))
                if (dp2 > 1)
                    cmds.push(gpu.computePass({ pipe:centroid, dispatch:1, binds: { meshes:meshbind, centroidwork }}))
            }
            cmds.push(gpu.computePass({ pipe:shapematch_init, dispatch:dp1, binds:{ meshes:meshbind, particles:bufs.particles, shapework }}))
            cmds.push(gpu.computePass({ pipe:shapematch, dispatch:dp1, binds: { meshes:meshbind, shapework }}))
            if (dp1 > 1) {
                cmds.push(gpu.computePass({ pipe:shapematch, dispatch:dp2, binds:{ meshes:meshbind, shapework }}))
                if (dp2 > 1)
                    cmds.push(gpu.computePass({ pipe:shapematch, dispatch:1, binds:{ meshes:meshbind, shapework }}))
            }
        }
        cmds.push(
            gpu.timestamp('shapematch'),
            gpu.computePass({ pipe:grads, dispatch:pd, binds:{ particles:bufs.particles, meshes:bufs.meshes }}),
            gpu.timestamp('gradients'),
            gpu.computePass({ pipe:collisions, dispatch:pd, binds:{ particles:bufs.particles, params:bufs.params, meshes:bufs.meshes } }),
            gpu.computePass({ pipe:collisions, dispatch:pd, binds:{ particles:bufs.particles, params:bufs.params, meshes:bufs.meshes } }),
            gpu.computePass({ pipe:collisions, dispatch:pd, binds:{ particles:bufs.particles, params:bufs.params, meshes:bufs.meshes } }),
            gpu.timestamp('stabilize'),
            gpu.computePass({ pipe:project, dispatch:pd, binds:{ particles:bufs.particles, meshes:bufs.meshes, params:bufs.params } }),
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
        const { gpu, particles, bufs, meshes, possessed, params, camPos } = this.sim
        if (particles.length == 0) return
        if (!phys.paused || this.fwdstep) {
            for (const idx of possessed) {
                let buf = gpu.chop(gpu.offset(bufs.meshes, Meshes.stride * idx), Mesh.size)
                gpu.write(buf, meshes[idx])
            }
            params.fcol = phys.collision
            params.shape = phys.shape
            params.friction = phys.friction
            params.spring = phys.spring
            params.t = tstep
            params.camPos = camPos
            params.ground = render.ground ? 1 : 0
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
        const { ctx, gpu, meshes, bufs, ntris, tris, bitmaps, particles, verts, camera, lights, possessed, vd, td, width, height } = this.sim

        gpu.configure(ctx, width, height, render)
        render.watch(render.keys.filter(key => key != 'fov'), () => { this.reset = true })
        
        camera.r = phys.d/2
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

        camera.t = performance.now()/1000
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
        //globalThis.meshes = new Meshes(await gpu.read(bufs.meshes))
        
            
        
    }

}














