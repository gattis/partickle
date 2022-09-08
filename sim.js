const module = globalThis
const { cos, sin, acos, asin, cbrt, sqrt, PI, random, ceil, floor, tan, max, min, log2,round,atan } = Math
import './gpu.js'
import './geometry.js'
import './ico80.js'

const D = 0.05
const FPS = 180
const T = 1/FPS
const SPEED = 1
const FEXT = Vec3.of(0, 0, -9.8)
const MAXNN = 24
const MAXEDGES = 18
const FCOL = 0.5
const FSHAPE = 0.01
const FRICTION = 1.5

const FOV = 70
let UP = Vec3.of(0,0,1)

const RED = Vec4.of(0.8, 0.2, 0.4, 1.0)
const BLUE = Vec4.of(0.3, 0.55, 0.7, 1.0);
const GREEN = Vec4.of(0.1, 0.5, 0.2, 1.0);
const BLACK = Vec4.of(0.0, 0.0, 0.0, 1.0);
const CLEAR = Vec4.of(0.0, 0.0, 0.0, 0.0);

let camPos = Vec3.of(0, -2, 1.3)
let camLR = 0
let camUD = 0
const camFwd = () => Vec3.of(sin(camLR) * cos(camUD), cos(camLR) * cos(camUD), sin(camUD))


const QUINN = { url: 'quinn.obj', texUrl: 'quinn.png',
                offset: Vec3.of(0,0,0), scale: Vec3.of(2, 2, 2), fext: Vec3.of(0, 0, -5),
                color: Vec4.of(1,.9,.9,0.0), particleColor: RED,
                sample: true }
const HAND = { url: 'hand.obj', texUrl: 'hand.png', color: Vec4.of(.9,.9,.9,0), particleColor: GREEN,
               offset: camPos.sub(Vec3.of(.4,.6,.3)), scale: Vec3.of(3.5,3.5,3.5),
               sample: true, possess: true, fext: Vec3.of(0, 0, 0) }

const TORUS = { url: 'torus.obj', offset: Vec3.of(0,0,1), scale: Vec3.of(1,1,1),
                color: CLEAR, particleColor: RED, fext: Vec3.of(0, 0, -5),
                sample: true }

const GROUND = { url: 'ground.obj', texUrl: 'marble.png',
                 fext: Vec3.of(0,0,0),
                 particleColor: CLEAR, color: Vec4.of(1,1,1,1),
                 sample:true, lock:()=>0.9 }



const CUBE = { url: 'cube.obj', sample: true, color: Vec4.of(.1,.5,.2,0.75) }

const WALL = { url: 'wall.obj', fext: Vec3.of(0,0,0), sample: true, color: Vec4.of(.1,.1,.1,.2), lock:()=>0.9 }

/*const WALL = { url: CUBE.url, fext: Vec3.of(0,0,0), sample: true, color: Vec4.of(.1,.1,.1,.2), lock:()=>0.9 }
const WALLS = [{ ...WALL, scale:Vec3.of(3,0.08,2), offset:Vec3.of(0.0,-1.46,1) },
               { ...WALL, scale:Vec3.of(3,0.08,2), offset:Vec3.of(0.0,1.46,1) },
               { ...WALL, scale:Vec3.of(0.08,3-.16,2), offset:Vec3.of(-1.46,0,1) },
               { ...WALL, scale:Vec3.of(0.08,3-.16,2), offset:Vec3.of(1.46,0,1) }]*/

const TETRA = { url: 'tetra.obj', fext: Vec3.of(0,0,0),
                particleColor: CLEAR, color: Vec4.of(1,1,1,1), fshape: 0,
                sample:true, lock:()=>0.9 }

const KNOT = { url: 'knot.obj', color: Vec4.of(.5,.5,.5,1), offset: Vec3.of(0,0,1),
               particleColor: Vec4.of(.5,.5,.5,0.3), sample:true, fext:Vec3.of(0)}

const HELPER = { url: 'helper.obj', scale: Vec3.of(2,2,2), sample: false, fext: Vec3.of(0) }

const MESHES = [
    KNOT,
    GROUND,
    WALL,
    
]

const clock = () => SPEED*performance.now()/1000

module.Mesh = GPU.struct({
    name: 'Mesh',
    fields: [
        ['c0', Vec3],
        ['ci', Vec3],
        ['color', Vec4],
        ['pcolor', Vec4],
        ['fext', Vec3],
        ['tex', i32],
        ['pi', u32],
        ['pf', u32],
        ['rot', Mat3],
        ['fshape', f32],
        ['padding', GPU.array({ type: u32, length: 27 })]        
    ]
})

module.Vertex = GPU.struct({
    name: 'Vertex',
    fields: [
        ['pos', Vec3],
        ['q', Vec3],
        ['mesh', u32],
        ['norm', Vec3],
        ['nedges', u32],
        ['particle', i32],
        ['edges', GPU.array({ type: u32, length: MAXEDGES })]
    ]
})
        

module.Particle = GPU.struct({
    name: 'Particle',
    fields: [
        ['sp', Vec3],
        ['hash', i32],
        ['si', Vec3],
        ['mesh', u32],
        ['v', Vec3],
        ['k', u32],
        ['q', Vec3],
        ['lock',f32],
        ['s0', Vec3],
        ['grad', Vec3],
        ['nn', GPU.array({ type: u32, length: MAXNN })],
    ]
})


module.Camera = GPU.struct({
    name: 'Camera',
    fields: [
        ['projection', Mat4],
        ['modelview', Mat4],
        ['pos', Vec3],
        ['selection', i32],
        ['forward', Vec3],
        ['d', f32]
    ]
})

module.Light = GPU.struct({
    name: 'Light',
    fields: [
        ['pos', Vec3],
        ['dir', Vec3],
        ['power', f32],
        ['color', Vec3]
    ]
})

module.Params = GPU.struct({
    name: 'Params',
    fields: [
        ['fcol', f32],
        ['fshape', f32],
        ['friction', f32],
    ]
})


module.TriVert = GPU.struct({
    name: 'TriVert',
    fields: [
        ['pos', Vec3],
        ['vidx', u32],
        ['norm', Vec3],
        ['mesh', u32],
        ['uv', Vec2],        
    ]
})
module.Triangle = GPU.array({ type: TriVert, length: 3 })


module.Meshes = GPU.array({ type: Mesh })
module.Vertices = GPU.array({ type: Vertex })
module.Particles = GPU.array({ type: Particle })
module.Triangles = GPU.array({ type: Triangle })
module.Mat3Array = GPU.array({ type: Mat3 })
module.Vec3Array = GPU.array({ type: Vec3 })


module.Sim = class Sim {

    async init(canvas) {
        const gpu = new GPU(canvas)
        await gpu.init()

        const objs = []
        for (const opt of MESHES) {
            const obj = await this.loadObj(opt)
            if (obj.sample) {
                const {samples, gradients} = voxelize(obj.verts, obj.faces, D)
                obj.virts = samples
                obj.virtgrads = gradients
            }
            objs.push(obj)
        }
        const nverts = objs.sum(obj => obj.verts.length)
        const nparticles = objs.sum(obj => obj.sample ? obj.virts.length : obj.verts.length)
        const ntris = objs.sum(obj => obj.faces.length)
        const meshes = Meshes.alloc(objs.length)
        const verts = Vertices.alloc(nverts)
        const particles = Particles.alloc(nparticles)
        const tris = Triangles.alloc(ntris)
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
            mesh.fext = obj.fext
            mesh.tex = -1
            mesh.rot = Mat3.of([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            mesh.fshape = obj.fshape
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
                v.particle = obj.sample ? -1 : cnts.particles + i
            }
            for (const [i,pos] of enumerate(obj.sample ? obj.virts : obj.verts)) {
                const p = particles[cnts.particles++]
                p.si = p.s0 = pos
                p.mesh = meshidx
                p.lock = obj.lock(pos)
                if (obj.virtgrads) p.grad = obj.virtgrads[i]
            }
            mesh.pf = cnts.particles
            for (const tri of obj.faces) {
                for (const i of range(3)) tri[i].vidx += vertoff
                tris[cnts.tris++] = tri
                vertedges[tri[0].vidx].push([tri[2].vidx,tri[1].vidx])
                vertedges[tri[1].vidx].push([tri[0].vidx,tri[2].vidx])
                vertedges[tri[2].vidx].push([tri[1].vidx,tri[0].vidx])
            }
        }
       
        for (const v of range(verts.length)) {
            const unsorted = vertedges[v], sorted = []
            if (unsorted.length > MAXEDGES) throw new Error(`meshes must have <= ${MAXEDGES} edges/vertex`)
            if (unsorted.length == 0) continue
            let first = unsorted.findIndex(e1 => unsorted.findIndex(e2 => e1[0]==e2[1]) == -1)
            if (first == -1) first = 0           // edge list is a cycle (mesh not open here)
            let nexti = unsorted[first][0]
            while (unsorted.length > 0) {
                sorted.push(nexti)
                const found = unsorted.findIndex(([i,f]) => i == nexti)
                if (found == -1) break
                nexti = unsorted[found][1]
                unsorted.splice(found, 1)
            }
            verts[v].nedges = sorted.length
            for (let e of range(sorted.length))
                verts[v].edges[e] = sorted[e]
        }

        for (const mesh of meshes) {
            let c = Vec3.of(0)
            for (let p = mesh.pi; p < mesh.pf; p++)
                c = c.add(particles[p].si)
            mesh.c0 = mesh.ci = c.divc(mesh.pf - mesh.pi)
        }
        for (const part of particles) 
            part.q = part.si.sub(meshes[part.mesh].c0)
        for (const vert of verts)
            vert.q = vert.pos.sub(meshes[vert.mesh].c0)
            

        const params = Params.alloc()
        params.fcol = FCOL
        params.fshape = FSHAPE
        params.friction = FRICTION

        const camera = Camera.alloc()
        const lights = GPU.array({ type: Light, length: 3 }).alloc(3)
        lights[0].color.x = lights[1].color.y = lights[2].color.z = 1
        lights[0].pos.z = lights[1].pos.z = lights[2].pos.z = 2
        lights[0].power = lights[1].power = lights[2].power = 5

        const threads = gpu.threads
        const pd = ceil(particles.length/gpu.threads)
        const vd = ceil(verts.length/gpu.threads)
        const td = ceil(tris.length/gpu.threads)
        
        const bufs = {
            particles: gpu.buf({ data:particles, usage: 'STORAGE|COPY_SRC|VERTEX' }),
            vertices: gpu.buf({ data:verts, usage: 'STORAGE|VERTEX|COPY_SRC' }),
            meshes: gpu.buf({ data:meshes, usage: 'STORAGE|COPY_DST|COPY_SRC' }),
            camera: gpu.buf({ data:camera, usage: 'UNIFORM|COPY_DST' }),
            params: gpu.buf({ data:params, usage: 'UNIFORM|COPY_DST' }),
            tris: gpu.buf({ data:tris, usage: 'STORAGE|VERTEX|COPY_SRC|COPY_DST' }),
            cnts: gpu.buf({ type:GPU.array({ type:i32, length: threads**3 }), usage: 'STORAGE|COPY_DST' }),
            work1: gpu.buf({ type:GPU.array({ type:i32, length: threads**2 }), usage: 'STORAGE' }),
            work2: gpu.buf({ type:GPU.array({ type:i32, length: threads }), usage: 'STORAGE' }),
            sorted: gpu.buf({ type:GPU.array({ type:u32, length: particles.length}), usage: 'STORAGE' }),
            lights: gpu.buf({ data: lights, usage: 'UNIFORM|FRAGMENT|COPY_DST' })
        }
        Object.assign(this, {gpu, objs, meshes, verts, particles, tris, bitmaps, camera, lights, bufs, possessed, canvas, params, pd, vd, td})

        this.compute = new Compute(this)
        this.render = new Render(this)

        await this.compute.setup()
        await this.render.setup()

    }
    
    async loadObj(opt) {
        const obj = {
            offset: Vec3.of(0), scale: Vec3.of(1),            
            color: Vec4.of(1), particleColor: CLEAR,
            sample: false, fext: FEXT, lock: ()=>0, fshape: 1.0
        }
        Object.assign(obj, opt)
        const data = await (await fetch(obj.url)).text()
        const lines = data.split(/[\r\n]/).map(line => line.split(/\s/))
        obj.verts = lines.filter(l=>l[0] == 'v').map(ts => {
            let coords = Vec3.of(...ts.slice(1,4).map(parseFloat))
            return coords.mul(obj.scale).add(obj.offset)
        })
        obj.virts = []
        const tex = lines.filter(l=>l[0] == 'vt').map(toks => Vec2.of(...toks.slice(1,3).map(parseFloat)))
        obj.faces = lines.filter(l=>l[0] == 'f').map(toks => Triangle.of(toks.slice(1).map(tok => {
            const [v,vt] = tok.split('/').slice(0,2).map(idx => parseInt(idx) - 1)
            return TriVert.of(Vec3.of(0), v, Vec3.of(0), 0, isNaN(vt) ? Vec2.of(0) : tex[vt])
        })))
        if (opt.texUrl) {
            const img = new Image()
            img.src = opt.texUrl
            await img.decode()
            obj.bitmap = await createImageBitmap(img)
        }
        return obj
    }

    run() {
        const { gpu, compute, render } = this
        const loop = async () => {
            if (!gpu.okay) return;
            await render.step()
            await compute.step()
            await compute.step()
            await compute.step()
            requestAnimationFrame(loop);
        }
        requestAnimationFrame(loop);
    }
}


class Compute {
    constructor(sim) {
        this.sim = sim
    }

    async setup() {
        const {gpu, verts, particles, meshes, params, bufs, pd, vd, td} = this.sim
        const threads = gpu.threads
        
        const wgsl = (await fetchtext('./compute.wgsl')).interp({threads, MAXNN, T, D})

        const shader = gpu.shader({ compute: true, wgsl: wgsl, defs: [Vertex, Particle, Mesh, Params, TriVert],
                                    storage: { particles:Particles, meshes:Meshes, vertices:Vertices, sorted:u32array, centroidwork:Vec3Array,
                                               cnts:i32array, cnts_atomic:iatomicarray, work:i32array, shapework:Mat3Array, tris:Triangles },
                                    uniform: { params:Params } })
        const predict = gpu.computePipe({ shader, entryPoint:'predict', binds:['particles','meshes','params']})
        const cntsort_cnt = gpu.computePipe({ shader, entryPoint:'cntsort_cnt', binds: ['particles','cnts_atomic'] })
        const prefsum_down = gpu.computePipe({ shader, entryPoint:'prefsum_down', binds: ['cnts','work'] })
        const prefsum_up = gpu.computePipe({ shader, entryPoint:'prefsum_up', binds: ['cnts','work'] })
        const cntsort_sort = gpu.computePipe({ shader, entryPoint:'cntsort_sort', binds: ['particles','cnts_atomic','sorted'] })
        const grid_collide = gpu.computePipe({ shader, entryPoint:'grid_collide', binds: ['particles','cnts','sorted'] })
        const collisions = gpu.computePipe({ shader, entryPoint:'collisions', binds: ['particles','params'] })
        const centroid_init = gpu.computePipe({ shader, entryPoint:'centroid_init', binds: ['meshes','particles','centroidwork'] })
        const centroid = gpu.computePipe({ shader, entryPoint:'centroid', binds: ['meshes','centroidwork'] })
        const shapematch_init = gpu.computePipe({ shader, entryPoint:'shapematch_init', binds: ['meshes','particles','shapework'] })
        const shapematch = gpu.computePipe({ shader, entryPoint:'shapematch', binds: ['meshes','shapework'] })
        const project = gpu.computePipe({ shader, entryPoint:'project', binds: ['particles','meshes','params'] })
        const vertpos = gpu.computePipe({ shader, entryPoint:'vertpos', binds: ['vertices','particles','meshes'] })
        const normals = gpu.computePipe({ shader, entryPoint:'normals', binds: ['vertices','particles'] })
        const sort_tris = gpu.computePipe({ shader, entryPoint:'sort_tris', binds: ['tris','vertices'] })

        const shapestage = []
        for (const [i,m] of enumerate(meshes)) {
            let n = m.pf - m.pi
            const centroidwork = gpu.buf({ type:Vec3Array, size:Vec3Array.stride*n, usage: 'STORAGE' })
            const shapework = gpu.buf({ type:Mat3Array, size: Mat3Array.stride*n, usage: 'STORAGE' })
            let dp1 = ceil(n/threads), dp2 = ceil(dp1/threads)
            const meshbind = gpu.offset(bufs.meshes, Meshes.stride*i)
            shapestage.push(gpu.computePass({ pipe:centroid_init, dispatch:dp1, binds: { meshes:meshbind, particles:bufs.particles, centroidwork }}))
            shapestage.push(gpu.computePass({ pipe:centroid, dispatch:dp1, binds: { meshes:meshbind, centroidwork }}))
            if (dp1 > 1) {
                shapestage.push(gpu.computePass({ pipe:centroid, dispatch:dp2, binds: { meshes:meshbind, centroidwork }}))
                if (dp2 > 1)
                    shapestage.push(gpu.computePass({ pipe:centroid, dispatch:1, binds: { meshes:meshbind, centroidwork }}))
            }
            shapestage.push(gpu.computePass({ pipe:shapematch_init, dispatch:dp1, binds:{ meshes:meshbind, particles:bufs.particles, shapework }}))
            shapestage.push(gpu.computePass({ pipe:shapematch, dispatch:dp1, binds: { meshes:meshbind, shapework }}))
            if (dp1 > 1) {
                shapestage.push(gpu.computePass({ pipe:shapematch, dispatch:dp2, binds:{ meshes:meshbind, shapework }}))
                if (dp2 > 1)
                    shapestage.push(gpu.computePass({ pipe:shapematch, dispatch:1, binds:{ meshes:meshbind, shapework }}))
            }
        }


        console.log(`nparts=${particles.length} nverts=${verts.length} threads=${gpu.threads} pd=${pd}`)
        this.batch = gpu.encode([
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
            gpu.timestamp('find collisions'),
            gpu.computePass({ pipe:collisions, dispatch:pd, binds:{ particles:bufs.particles, params:bufs.params } }),
            gpu.computePass({ pipe:collisions, dispatch:pd, binds:{ particles:bufs.particles, params:bufs.params } }),
            gpu.computePass({ pipe:collisions, dispatch:pd, binds:{ particles:bufs.particles, params:bufs.params } }),
            gpu.timestamp('stabilize collisions'),
            ...shapestage,
            gpu.timestamp('shapematch'),
            gpu.computePass({ pipe:project, dispatch:pd, binds:{ particles:bufs.particles, meshes:bufs.meshes, params:bufs.params } }),
            gpu.timestamp('project'),
            gpu.computePass({ pipe:vertpos, dispatch:vd, binds:{ vertices:bufs.vertices, particles:bufs.particles, meshes:bufs.meshes} }),
            gpu.timestamp('vertexpositions'),
            gpu.computePass({ pipe:normals, dispatch:vd, binds:{ vertices:bufs.vertices, particles:bufs.particles } }),
            gpu.timestamp('vertexnormals'),
            gpu.computePass({ pipe:sort_tris, dispatch:td, binds:{ vertices:bufs.vertices, tris:bufs.tris} }),
            gpu.timestamp('sort_tris'),
        ])
        
        this.fwdstep = false
        this.tstart = this.tsim = this.tlast = clock()
        this.frames = 0

    }

    async stats() {
        const data = new BigInt64Array(await this.sim.gpu.read(this.batch.stampBuf))
        const labels = this.batch.stampLabels
        const ret = { stamps: data, labels, tstart: this.tstart, tlast: this.tlast, frames: this.frames, tsim: this.tsim }
        this.tstart = this.tlast
        this.frames = 0
        return ret
    }
        
    get paused() {
        return localStorage.paused == 'true'
    }

    set paused(val) {        
        localStorage.paused = val ? 'true' : ''
    }

    async step() {
        const { gpu, bufs, meshes, possessed, params } = this.sim
        if (!this.paused || this.fwdstep) {
            for (const idx of possessed) {
                gpu.write({ buf:bufs.meshes, data: meshes, arrayIdx: idx, field: Mesh.ci})
                gpu.write({ buf:bufs.meshes, data: meshes, arrayIdx: idx, field: Mesh.rot})
            }
            params.fcol = localStorage.fcol == undefined ? FCOL : parseFloat(localStorage.fcol)
            params.fshape = localStorage.fshape == undefined ? FSHAPE : parseFloat(localStorage.fshape)
            params.friction = localStorage.friction == undefined ? FRICTION : parseFloat(localStorage.friction)
            gpu.write({ buf:bufs.params, data: params })
            this.batch.execute()
        }
        this.tsim += T
        this.tlast = clock()
        this.frames++        
        


        if (this.fwdstep) {
            gpu.read(bufs.vertices).then(buf => {
                module.verts = new Vertices(buf)
                gpu.read(bufs.tris).then(buf => {
                    module.tris = new Triangles(buf)
                })
            })
            
            this.fwdstep = false        
        }
        
    }
    
}



class Render {
    constructor(sim) {
        this.sim = sim
    }

    async setup() {
        const { gpu, meshes, bufs, tris, bitmaps, particles, verts, camera, lights, possessed } = this.sim
        
        camera.d = D
        camera.selection = -1

        const partDraws = partMesh.length
        for (const i of range(partDraws))
            partMesh[i] = partMesh[i].mulc(D/2)
        const partWgsl = partMesh.map(v=>`v3(${v.x}, ${v.y}, ${v.z})`).join(',')
        let wgsl = (await fetchtext('./render.wgsl'))
        wgsl = wgsl.interp({partWgsl, partDraws, numLights: lights.length })
        const shader = gpu.shader({ wgsl, defs:[Vertex, Particle, Mesh, Camera, Light],
                                    storage:{ particles:Particles, meshes:Meshes, vertices:Vertices },
                                    uniform:{ camera:Camera, lights:lights.constructor },
                                    textures:{ tex:{ name:'texture_2d_array<f32>' } },
                                    samplers:{ samp:{ name:'sampler' } } })
        

        let showParticles = false
        for (const mesh of meshes)
            if (mesh.pcolor[3] > 0)
                showParticles = true

        console.log('partPipe')
        const partPipe = gpu.renderPipe({
            shader, vert:'vert_part', frag:'frag_part', binds: ['meshes', 'camera', 'lights'],
            vertBufs: [{ buf:bufs.particles, arrayStride:Particles.stride, stepMode: 'instance',
                         attributes: [{ shaderLocation:0, offset:Particle.si.off, format:'float32x3' },
                                      { shaderLocation:1, offset:Particle.mesh.off, format:'uint32' }]}]
        })
        
        const surfPipeDesc = {
            shader, vert:'vert_surf', binds: ['meshes', 'camera', 'lights', 'tex', 'samp'],
            vertBufs: [{ buf:bufs.tris, arrayStride:Triangle.stride,
                         attributes: [{ shaderLocation:0, offset:TriVert.pos.off, format:'float32x3' },
                                      { shaderLocation:1, offset:TriVert.norm.off, format:'float32x3' },
                                      { shaderLocation:2, offset:TriVert.mesh.off, format:'uint32' },
                                      { shaderLocation:3, offset:TriVert.uv.off, format:'float32x2' }]}]
        }
        console.log('surfPipeOpaque')
        const surfPipeOpaque = gpu.renderPipe({ ...surfPipeDesc, frag:'frag_surf_opaque' })
        console.log('surfPipeTransp')
        const surfPipeTransp = gpu.renderPipe({ ...surfPipeDesc, frag:'frag_surf_transp', depthWriteEnabled: false,
                                                blend: { color: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
                                                         alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' }}})
        console.log('normPipe')
        const normPipe = gpu.renderPipe({
            shader, vert:'vert_norm', frag:'frag_norm', binds: ['camera'], topology: 'line-list',
            vertBufs: [{ buf:bufs.tris, arrayStride:Triangle.stride, stepMode: 'instance',
                         attributes: [{ shaderLocation:0, offset:TriVert.pos.off, format:'float32x3' },
                                      { shaderLocation:1, offset:TriVert.norm.off, format:'float32x3' }]}]
        })



        let tex = gpu.texture(bitmaps)
        let samp = gpu.sampler()       
        const binds = {meshes:bufs.meshes, camera:bufs.camera, lights:bufs.lights}
        const draws = []
        draws.push(gpu.draw({ pipe:surfPipeOpaque, dispatch:tris.length*3, binds:{ ...binds, tex, samp }}))
        draws.push(gpu.draw({ pipe:normPipe, dispatch:[2, tris.length*3], binds:{ camera:bufs.camera }}))
        if (showParticles)
            draws.push(gpu.draw({ pipe:partPipe, dispatch:[partDraws, particles.length], binds }))
        draws.push(gpu.draw({ pipe:surfPipeTransp, dispatch:tris.length*3, binds:{ ...binds, tex, samp }}))

        
        this.batch = gpu.encode([gpu.renderPass(draws)])

        this.tstart = this.tlast = clock()
        this.frames = 0
    }

    async stats() {
        const ret = { tstart: this.tstart, tlast: this.tlast, frames: this.frames }
        this.tstart = this.tlast
        this.frames = 0
        return ret
    }
    
    async step() {
        const { camera, lights, canvas, gpu, bufs } = this.sim
        camera.pos = camPos
        camera.projection = Mat4.perspective(FOV, canvas.width/canvas.height, .01, 200)
        camera.forward = camFwd()
        camera.modelview = Mat4.look(camPos, camera.forward, UP)
        gpu.write({ buf:bufs.camera, data: camera})
        let offset = 0
        for (const i of range(3)) {
            const theta = offset + performance.now()/1000
            lights[i].pos.x = 2*cos(theta)
            lights[i].pos.y = 2*sin(theta)
            offset += 2*PI/3
        }
        gpu.write({ buf:bufs.lights, data: lights})

        /*await gpu.dev.queue.onSubmittedWorkDone()
        const tris = new Triangles(await gpu.read(bufs.tris))
        const triArr = []
        for (const tri of tris)
            triArr.push(tri)
        triArr.sort((a,b) => {
            let ca = Vec3.of(0), cb = Vec3.of(0)
            for (const i of range(3)) {
                ca = ca.add(a[i].pos)
                cb = cb.add(b[i].pos)
            }
            ca = ca.divc(3)
            cb = cb.divc(3)
            return cb.dist(camPos) - ca.dist(camPos)
        })
        for (const i of range(tris.length))
            tris[i] = triArr[i]
        gpu.write({ buf:bufs.tris, data: tris })
        await gpu.dev.queue.onSubmittedWorkDone()*/
        
        this.batch.execute()
        this.frames++
        this.tlast = clock()
    }

}


class App {
    constructor() {
        this.canvas = document.createElement('canvas')
        document.body.append(this.canvas)
        this.canvas.width = this.canvas.style.width = window.innerWidth
        this.canvas.height = this.canvas.style.height = window.innerHeight
        this.sim = new Sim(this)
        this.sim.init(this.canvas).then(()=> {
            this.setup()
            this.sim.run()
        })        
    }


    setup() {
        const { render, compute, possessed, params, particles } = this.sim
        const cv = this.canvas
        
        const nav = document.createElement('div')
        const info = document.createElement('div')
        const ctrls = document.createElement('div')
        const pause = document.createElement('button')
        const step = document.createElement('button')
        const inspect = document.createElement('div')
        const pedit = document.createElement('div')           

        const commonStyle = {fontFamily: 'monospace', position: 'fixed', color: 'white'}
        Object.assign(ctrls.style, { left:5, top:5, ...commonStyle })
        Object.assign(pedit.style, { left:5, top:35, textAlign:'right', ...commonStyle })
        Object.assign(info.style, { right:5, top:25, textAlign:'right', ...commonStyle })
        Object.assign(nav.style, { right:5, top:5, ...commonStyle })
        Object.assign(inspect.style, { marginLeft: 20 })
        
        pause.innerHTML = "&#9199;"
        step.innerHTML = "&#11122;"
        step.style.display = inspect.style.display = localStorage.paused ? 'inline' : 'none'
        ctrls.append(pause,step,inspect)
        cv.parentNode.append(ctrls,info)
        cv.parentNode.append(nav)
        cv.parentNode.append(pedit)

        inspect.innerHTML = `particle: <input id='partidx' type='number' min='0' max='${particles.length - 1}' value='0'>`
        const partidx = document.querySelector('#partidx')
        Object.assign(partidx.style, { width: 70, textAlign: 'right' })
        
        pedit.innerHTML = Params.fields.map(field => {
            const name = field.name
            const val = localStorage[name] == undefined ? params[name] : parseFloat(localStorage[name])
            const lo = params[name] * 0.002, hi = params[name] * 4.9
            return `${name} <input style='vertical-align:middle;width:200px' type='range' name='${name}' value='${val}' `+
                `min='${lo}' max='${hi}' step='0.001' ` +
                `oninput='${name}.value=parseFloat(this.value).toFixed(3); localStorage.${name} = this.value'/> ` +
                `<output id='${name}'>${val.toFixed(3)}</output>`
        }).join('<br/>')
                                           

        this.move = false
        cv.onmouseup = cv.onmouseout = cv.onmousedown = cv.onmousemove = cv.onwheel = cv.oncontextmenu =
            pause.onclick = step.onclick = window.onresize = partidx.oninput = (ev) => { this.handle(ev) }

        setTimeout(()=>{ this.updateInfo() }, 500)
        
        Object.assign(this, {nav, info, ctrls, pause, step, inspect, partidx })
    }

    handle(ev) {
        const { compute, render, gpu, bufs, possessed, camera } = this.sim
        if (ev.target == this.pause && ev.type == 'click') {
            compute.paused = !compute.paused
            this.step.style.display = this.inspect.style.display = compute.paused ? 'inline' : 'none'
        }
    
        if (ev.target == this.step && ev.type == 'click') {
            compute.fwdstep = true
        }

        if (ev.target == this.partidx && ev.type == 'input') {
            const pidx = parseInt(this.partidx.value)
            camera.selection = pidx
            gpu.read(bufs.particles).then(buf => {
                const particles = new Particles(buf)
                const p = particles[pidx]
                console.clear()
                console.log({particle: pidx, si: p.si.toString(), grad: p.grad.toString()})               
            })        
        }

        if (ev.type == 'mousedown') {
            this.move = {x: ev.x, y: ev.y, btn: ev.button}
        }

        if (ev.type == 'mouseout' || ev.type == 'mouseup') {
            this.move = false;
        }
        
        if (ev.type == 'mousemove') {
            if (!this.move) return
            const dx = .01*(ev.x - this.move.x), dy = .01*(this.move.y - ev.y)
            this.move.x = ev.x
            this.move.y = ev.y
            if (this.move.btn == 0) {
                camLR += dx
                camUD = clamp(camUD + dy,-PI/2, PI/2)
            } else if (this.move.btn == 1) {
                const delta = Vec3.of( -dx * cos(camLR), dx * sin(camLR), -dy);
                camPos = camPos.add(delta)
            }
        }

        if (ev.type == 'wheel') {
            const dy = -.001 * ev.deltaY
            const camDir = camFwd()
            camPos.x += dy * camDir.x
            camPos.y += dy * camDir.y
            camPos.z += dy * camDir.z
        }

        if (ev.type == 'contextmenu') {
            ev.preventDefault()
        }

        if (ev.type == 'resize') {
            this.canvas.width = this.canvas.style.width = window.innerWidth
            this.canvas.height = this.canvas.style.height = window.innerHeight
            gpu.resize()
        }
            
    }
    
    async updateInfo() {
        const {compute, render} = this.sim
        let { tstart, tlast, frames, stamps, labels, tsim } = await compute.stats()
        const profile = Array.from(range(1,labels.length)).map(i => `${labels[i]}: ${(stamps[i] - stamps[i-1])/1000n} &mu;s`)
        if (labels.length) profile.push(`total: ${(stamps[labels.length-1] - stamps[0])/1000n} &mu;s`)
        const trel = `compute trel:${((tlast - tsim)*1000).toFixed(2)} ms`
        const computeFps = `compute fps:${(frames/(tlast-tstart)).toFixed(2)}`
        ;({ tstart, tlast, frames } = await render.stats())
        const renderFps = `render fps:${(frames/(tlast-tstart)).toFixed(2)}`
        this.info.innerHTML = [trel, computeFps, renderFps, ...profile].join('<br/>')
        this.nav.innerText = `cam: pos=${camPos.toString()} fwd=${camFwd().toString()}`
        setTimeout(() => { this.updateInfo() }, 500)
    }
    
}



module.app = new App()









