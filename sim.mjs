const { cos, sin, acos, asin, cbrt, sqrt, PI, random, ceil, floor, tan, max, min, log2,round,atan } = Math
import * as util from './utils.mjs'
import * as gpu from './gpu.mjs'
import * as geo from './geometry.mjs'
Object.assign(globalThis, util, gpu, geo)


const D = 0.05
const FRAMERATIO = 3
const SPEED = 1
const FEXT = v3(0, 0, -9.8)
const MAXNN = 24
const MAXEDGES = 18
const FCOL = 0.5
const FSHAPE = 0.0
const FRICTION = 1.5

const FOV = 60
let UP = v3(0,0,1)
let CAM_POS = v3(0, -4, 2)

const RED = v4(0.8, 0.2, 0.4, 1.0)
const BLUE = v4(0.3, 0.55, 0.7, 1.0);
const GREEN = v4(0.1, 0.5, 0.2, 1.0);
const BLACK = v4(0.0, 0.0, 0.0, 1.0);
const CLEAR = v4(0.0, 0.0, 0.0, 0.0);


const showNormals = false
const showGrads = false
const showAxes = false
const particleColor = v4(.3,.3,.8,1)

const QUINN = { url:'quinn.obj', texUrl:'quinn.png', offset:v3(0,0,0), sample:true,
                scale:v3(2, 2, 2), fext:v3(0, 0, -5), color:v4(1,.9,.9,0.0) }

const HAND = { url:'hand.obj', texUrl:'hand.png', color:v4(.9,.9,.9,0), 
               offset:v3(.4,.6,.3), scale:v3(3.5,3.5,3.5),
               sample:true, possess:true, fext:v3(0, 0, 0) }

const TORUS = { url:'torus.obj', offset:v3(0,0,1), scale:v3(1),
                color:RED, particleColor:v4(.7,.7,.1,1), sample:true, fext:v3(0,0,0) }

const GROUND = { url:'ground.obj', texUrl:'marble.png', fext:v3(0,0,0),
                 color:CLEAR, sample:true, lock:()=>0.99 }

const CUBE = { url:'cube.obj', sample:true, color:v4(.3,.3,.8,.7), fext:v3(0), scale:v3(1) }

const TRI = { url:'tri.obj', sample:true, color:v4(0.1,0.4,0.6,0.7), fext:v3(0) }

const WALL = { url:'wall.obj', fext:v3(0,0,0), sample:true, color:v4(.1,.1,.1,.2), lock:()=>0.9 }

const TETRA = { url:'tetra.obj', fext:v3(0,0,0), color:v4(1,1,1,1), fshape:0,
                sample:true, lock:()=>0.9 }

const KNOT = { url:'knot.obj', color:v4(.6,.3,.3,.5), offset:v3(0,0,1), sample:true, scale:v3(2)}

const HELPER = { url: 'helper.obj', scale: v3(2,2,2), sample: false, fext: v3(0) }

const MESHES = [
    //GROUND
    //WALL
    TORUS
    //CUBE,
    //{url:'particle.obj', sample:false, offset:v3(0,0,1)},
    //{url:'particle.obj', sample:false, offset:v3(.1,0,0), fext: v3(0)},
    
]

const clock = () => SPEED*performance.now()/1000

export const Mesh = GPU.struct({
    name: 'Mesh',
    fields: [
        ['c0', V3],
        ['tex', i32],
        ['ci', V3],
        ['pi', u32],
        ['pcolor', V4],
        ['pf', u32],
        ['fext', V3],
        ['fshape', f32],
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
        ['grad0', V3],
        ['grad', V3],
        ['nn', GPU.array({ type: u32, length: MAXNN })],
    ]
})


export const Camera = GPU.struct({
    name: 'Camera',
    fields: [
        ['projection', M4],
        ['modelview', M4],
        ['pos', V3],
        ['selection', i32],
        ['forward', V3],
        ['r', f32],
        ['ratio', f32],
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
        ['fcol', f32],
        ['fshape', f32],
        ['friction', f32],
        ['t',f32]
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
    ]
})
export const Triangle = GPU.array({ type: TriVert, length: 3 })


export const Meshes = GPU.array({ type: Mesh })
export const Vertices = GPU.array({ type: Vertex })
export const Particles = GPU.array({ type: Particle })
export const Triangles = GPU.array({ type: Triangle })
export const m3Array = GPU.array({ type: M3 })
export const V3Array = GPU.array({ type: V3 })


export class Sim {

    async init(width, height, ctx) {
        const gpu = new GPU()
        await gpu.init(width,height,ctx)

        this.camPos = CAM_POS
        this.camLR = 0
        this.camUD = -PI/6        
        
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
            mesh.rot = m3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            mesh.fshape = obj.fshape
            if (obj.possess)
                possessed.push(mesh)                
            if (obj.bitmap) {
                mesh.tex = bitmaps.length
                bitmaps.push(obj.bitmap)
            }
            let nn
            if (obj.sample) {
                //nn = new NN(obj.virts)
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
                if (obj.virtgrads) p.grad0 = p.grad = obj.virtgrads[i]
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
        params.fcol = FCOL
        params.fshape = FSHAPE
        params.friction = FRICTION

        const camera = Camera.alloc()
        const lights = GPU.array({ type: Light, length: 1 }).alloc(1)
        lights[0].color = v3(1,.85,.6); //lights[1].color.y = lights[2].color.z = 1
        lights[0].pos.z = 3; //lights[1].pos.z = lights[2].pos.z = 2
        lights[0].power = 5; //lights[1].power = lights[2].power = 5

        const threads = gpu.threads
        const pd = ceil(particles.length/gpu.threads)
        const vd = ceil(verts.length/gpu.threads)
        const td = ceil(tris.length/gpu.threads)
        
        const buflist = [
            gpu.buf({ label:'particles', data:particles, usage: 'STORAGE|COPY_SRC|VERTEX' }),
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
        Object.assign(this, {gpu, objs, meshes, verts, particles, tris, bitmaps, camera, lights, bufs, possessed, ctx, width, height, params, pd, vd, td})

        this.compute = new Compute(this)
        this.render = new Render(this)
        
        await this.compute.setup()
        await this.render.setup()

    }

    resize(width, height) {
        this.width = width
        this.height = height
        this.gpu.resize(width,height)
    }

    camFwd() {
        return v3(sin(this.camLR) * cos(this.camUD), cos(this.camLR) * cos(this.camUD), sin(this.camUD))
    }
    
    async loadObj(opt) {
        const obj = {
            offset: v3(0), scale: v3(1),            
            color: v4(1), particleColor,
            sample: false, fext: FEXT, lock: ()=>0, fshape: 1.0
        }
        Object.assign(obj, opt)
        const data = await (await fetch(obj.url)).text()
        const lines = data.split(/[\r\n]/).map(line => line.split(/\s/)).filter(l=>l[0] != '#')
        obj.verts = lines.filter(l=>l[0] == 'v').map(ts => {
            let coords = v3(...ts.slice(1,4).map(parseFloat))
            return coords.mul(obj.scale).add(obj.offset)
        })
        obj.virts = []
        const tex = lines.filter(l=>l[0] == 'vt').map(toks => v2(...toks.slice(1,3).map(parseFloat)))
        obj.faces = lines.filter(l=>l[0] == 'f').map(toks => Triangle.of(toks.slice(1).map(tok => {
            const [v,vt] = tok.split('/').slice(0,2).map(idx => parseInt(idx) - 1)
            return TriVert.of(v3(0), v, v3(0), 0, isNaN(vt) ? v2(0) : tex[vt])
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
        let lastStamp = null
        const loop = (stamp) => {
            if (!gpu.okay) return;
            requestAnimationFrame(loop);           
            render.step()
            if (lastStamp != null) {
                const tstep = (stamp - lastStamp) / 1000 / FRAMERATIO
                for (let i of range(FRAMERATIO))
                   compute.step(tstep)
            }
            lastStamp = stamp
        }
        requestAnimationFrame(loop);
    }
}


export class Compute {
    constructor(sim) {
        this.sim = sim
    }

    async setup() {
        const { gpu, verts, particles, meshes, params, bufs, pd, vd, td } = this.sim
        if (particles.length == 0) return
        const threads = gpu.threads
        const wgsl = (await fetchtext('./compute.wgsl')).interp({threads, MAXNN, D})

        const shader = await gpu.shader({ compute: true, wgsl: wgsl, defs: [Vertex, Particle, Mesh, Params, TriVert],
                                    storage: { particles:Particles, meshes:Meshes, vertices:Vertices, sorted:u32array, centroidwork:V3Array,
                                               cnts:i32array, cnts_atomic:iatomicarray, work:i32array, shapework:m3Array, tris:Triangles },
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
        const collisions = gpu.computePipe({ shader, entryPoint:'collisions', binds: ['particles','params'] })
        const project = gpu.computePipe({ shader, entryPoint:'project', binds: ['particles','meshes','params'] })
        const vertpos = gpu.computePipe({ shader, entryPoint:'vertpos', binds: ['vertices','particles','meshes'] })
        const normals = gpu.computePipe({ shader, entryPoint:'normals', binds: ['vertices','particles'] })
        const sort_tris = gpu.computePipe({ shader, entryPoint:'sort_tris', binds: ['tris','vertices'] })

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
            gpu.computePass({ pipe:collisions, dispatch:pd, binds:{ particles:bufs.particles, params:bufs.params } }),
            gpu.computePass({ pipe:collisions, dispatch:pd, binds:{ particles:bufs.particles, params:bufs.params } }),
            gpu.computePass({ pipe:collisions, dispatch:pd, binds:{ particles:bufs.particles, params:bufs.params } }),
            gpu.timestamp('stabilize collisions'),
            gpu.computePass({ pipe:project, dispatch:pd, binds:{ particles:bufs.particles, meshes:bufs.meshes, params:bufs.params } }),
            gpu.timestamp('project'),
            gpu.computePass({ pipe:vertpos, dispatch:vd, binds:{ vertices:bufs.vertices, particles:bufs.particles, meshes:bufs.meshes} }),
            gpu.timestamp('vertexpositions'),
            gpu.computePass({ pipe:normals, dispatch:vd, binds:{ vertices:bufs.vertices, particles:bufs.particles } }),
            gpu.timestamp('vertexnormals')
        )
        if (bufs.tris.size > 0)
            cmds.push(
                gpu.computePass({ pipe:sort_tris, dispatch:td, binds:{ vertices:bufs.vertices, tris:bufs.tris} }),
                gpu.timestamp('sort_tris')
            )
        
        this.batch = gpu.encode(cmds)
        this.fwdstep = false
        this.tstart = this.tsim = this.tlast = clock()
        this.frames = 0
        this.tsteps = []

    }

    async stats() {
        if (!this.batch) return {}
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

    step(tstep) {
        const { gpu, particles, bufs, meshes, possessed, params } = this.sim
        if (particles.length == 0) return

        if (this.tsteps.length > 10)
            this.tsteps.shift()
        this.tsteps.push(tstep)
        const tsmooth = this.tsteps.sum()/this.tsteps.length

        if (!this.paused || this.fwdstep) {
            for (const idx of possessed) {
                gpu.write({ buf:bufs.meshes, data: meshes, arrayIdx: idx, field: Mesh.ci})
                gpu.write({ buf:bufs.meshes, data: meshes, arrayIdx: idx, field: Mesh.rot})
            }
            params.fcol = localStorage.fcol == undefined ? FCOL : parseFloat(localStorage.fcol)
            params.fshape = localStorage.fshape == undefined ? FSHAPE : parseFloat(localStorage.fshape)
            params.friction = localStorage.friction == undefined ? FRICTION : parseFloat(localStorage.friction)


           
            params.t = tsmooth
            gpu.write({ buf:bufs.params, data: params })
            this.batch.execute()
        }
        this.tsim += tsmooth
        this.tlast = clock()
        this.frames++        
        
        if (this.fwdstep) {
            /*gpu.read(bufs.vertices).then(buf => {
                export const verts = new Vertices(buf)
                gpu.read(bufs.tris).then(buf => {
                    const tris = new Triangles(buf)
                })
            })*/
            
            this.fwdstep = false        
        }
        
    }
    
}



export class Render {
    constructor(sim) {
        this.sim = sim
    }

    async setup() {
        const { gpu, meshes, bufs, tris, bitmaps, particles, verts, camera, lights, possessed } = this.sim
        
        camera.r = D/2
        camera.selection = -1

        let wgsl = (await fetchtext('./render.wgsl'))
        wgsl = wgsl.interp({numLights: lights.length })
        const shader = await gpu.shader({ wgsl, defs:[Vertex, Particle, Mesh, Camera, Light],
                                    storage:{ particles:Particles, meshes:Meshes, vertices:Vertices },
                                    uniform:{ camera:Camera, lights:lights.constructor },
                                    textures:{ tex:{ name:'texture_2d_array<f32>' } },
                                    samplers:{ samp:{ name:'sampler' } } })       

        const transp = { depthWriteEnabled: false,
                         blend: { color: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
                                  alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } }}

        const partPipe = gpu.renderPipe({
            shader, vert:'vert_part', frag:'frag_part', binds: ['meshes', 'camera', 'lights'],
            vertBufs: [{ buf:bufs.particles, arrayStride:Particles.stride, stepMode: 'instance',
                         attributes: [{ shaderLocation:0, offset:Particle.si.off, format:'float32x3' },
                                      { shaderLocation:1, offset:Particle.mesh.off, format:'uint32' }]}] })
        const lightPipe = gpu.renderPipe({ shader, vert:'vert_light', frag:'frag_light', binds: ['camera','lights'], ...transp })
        
        const surfPipeDesc = {
            shader, vert:'vert_surf', binds: ['meshes', 'camera', 'lights', 'tex', 'samp'],
            vertBufs: [{ buf:bufs.tris, arrayStride:Triangle.stride,
                         attributes: [{ shaderLocation:0, offset:TriVert.pos.off, format:'float32x3' },
                                      { shaderLocation:1, offset:TriVert.norm.off, format:'float32x3' },
                                      { shaderLocation:2, offset:TriVert.mesh.off, format:'uint32' },
                                      { shaderLocation:3, offset:TriVert.uv.off, format:'float32x2' }]}]
        }
                             
        const surfPipeOpaque = gpu.renderPipe({ ...surfPipeDesc, frag:'frag_surf_opaque', cullMode:'none'})
        const surfPipeTransp = gpu.renderPipe({ ...surfPipeDesc, frag:'frag_surf_transp', ...transp, cullMode:'none' })

        const axisPipe = gpu.renderPipe({ shader, vert:'vert_axis', frag:'frag_axis', binds:['camera'], topology:'line-list' })
        const normPipe = gpu.renderPipe({
            shader, vert:'vert_norm', frag:'frag_norm', binds: ['camera'], topology: 'line-list',
            vertBufs: [{ buf:bufs.tris, arrayStride:Triangle.stride, stepMode: 'instance',
                         attributes: [{ shaderLocation:0, offset:TriVert.pos.off, format:'float32x3' },
                                      { shaderLocation:1, offset:TriVert.norm.off, format:'float32x3' }]}]
        })
        const gradPipe = gpu.renderPipe({
            shader, vert:'vert_norm', frag:'frag_norm', binds: ['camera'], topology: 'line-list',
            vertBufs: [{ buf:bufs.particles, arrayStride:Particles.stride, stepMode: 'instance',
                         attributes: [{ shaderLocation:0, offset:Particle.si.off, format:'float32x3' },
                                      { shaderLocation:1, offset:Particle.grad.off, format:'float32x3' }]}]
        })

        let tex = gpu.texture(bitmaps)
        let samp = gpu.sampler()       
        const draws = []
        
        draws.push(gpu.draw({ pipe:surfPipeOpaque, dispatch:tris.length*3,
                              binds:{ meshes: bufs.meshes, camera:bufs.camera, lights:bufs.lights, tex, samp }}))
        draws.push(gpu.draw({ pipe:partPipe, dispatch:[12, particles.length],
                              binds:{ meshes: bufs.meshes, camera:bufs.camera, lights:bufs.lights } }))

        if (showAxes)
            draws.push(gpu.draw({ pipe:axisPipe, dispatch:[2, 3], binds:{ camera:bufs.camera }}))
        if (showNormals)
            draws.push(gpu.draw({ pipe:normPipe, dispatch:[2, tris.length*3], binds:{ camera:bufs.camera }}))
        if (showGrads)
            draws.push(gpu.draw({ pipe:gradPipe, dispatch:[2, particles.length], binds:{ camera:bufs.camera }}))
        draws.push(gpu.draw({ pipe:lightPipe, dispatch:[3, lights.length],
                              binds: { camera:bufs.camera, lights:bufs.lights } }))
        draws.push(gpu.draw({ pipe:surfPipeTransp, dispatch:tris.length*3,
                              binds:{ meshes: bufs.meshes, camera:bufs.camera, lights:bufs.lights, tex, samp }}))

        
        
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
    
    step() {
        const { camera, lights, ctx, width, height, gpu, bufs, camPos } = this.sim
        camera.pos = camPos
        camera.ratio = width/height
        camera.projection = M4.perspective(FOV, camera.ratio, .01, 200)
        camera.forward = this.sim.camFwd()
        camera.modelview = M4.look(camPos, camera.forward, UP)
        gpu.write({ buf:bufs.camera, data: camera})
        
        /*for (const i of range(lights.length)) {
            const theta = offset + performance.now()/1000
            lights[i].pos.x = 2*cos(theta)
            lights[i].pos.y = 2*sin(theta)
            offset += 2*PI/3
        }*/
        gpu.write({ buf:bufs.lights, data: lights})

        /*await gpu.dev.queue.onSubmittedWorkDone()
        const tris = new Triangles(await gpu.read(bufs.tris))
        const triArr = []
        for (const tri of tris)
            triArr.push(tri)
        triArr.sort((a,b) => {
            let ca = v3(0), cb = v3(0)
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














