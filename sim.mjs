const { cos, sin, acos, asin, cbrt, sqrt, PI, random, ceil, floor, tan, max, min, log2,round,atan } = Math
import * as util from './utils.mjs'
import * as gpu from './gpu.mjs'
import * as geo from './geometry.mjs'
Object.assign(globalThis, util, gpu, geo)


const D = 0.01
const FRAMERATIO = 3
const SPEED = 1
const FEXT = v3(0, 0, -9.8)
const MAXNN = 24
const MAXEDGES = 18
const FCOL = 0.1
const FSHAPE = 0.01
const FRICTION = 1.5

const FOV = 60
let UP = v3(0,0,1)
let CAM_POS = v3(0, -4.5, 3)
let CAM_UD = -PI/7

const RED = v4(0.8, 0.2, 0.4, 1.0)
const BLUE = v4(0.3, 0.55, 0.7, 1.0);
const GREEN = v4(0.1, 0.5, 0.2, 1.0);
const BLACK = v4(0.0, 0.0, 0.0, 1.0);
const CLEAR = v4(0.0, 0.0, 0.0, 0.0);

export const renderPref = {
    normals: false,
    grads: false,
    axes: false,
    depth_write: true,
    atc: true,
    samples: 4,
    depth_compare: 'less-equal',
    cull: 'back',
    color_op:'add',
    alpha_op:'add',
    color_src:'one',
    color_dst:'one-minus-src-alpha',
    alpha_src:'one',
    alpha_dst:'one-minus-src-alpha',
    alpha_mode:'premultiplied',
    format: 'rgba8unorm',
    depth_format: 'depth32float'
}

const opOpts = ['add','subtract','reverse-subtract','min','max']
const factorOpts = ['zero','one','constant','one-minus-constant',
                    'src','one-minus-src','src-alpha','one-minus-src-alpha','src-alpha-saturated',
                    'dst','one-minus-dst','dst-alpha','one-minus-dst-alpha']
export const renderOpts = {
    depth_compare:['less-equal','less','greater-equal','greater','always','never'],
    cull:['none','back','front'],
    color_op:opOpts, alpha_op:opOpts,
    color_src:factorOpts, color_dst:factorOpts, alpha_src:factorOpts, alpha_dst:factorOpts,
    alpha_mode:['opaque','premultiplied'],
    format:['rgba8unorm','bgra8unorm','rgba16float'],
    depth_format:['depth16unorm','depth24plus','depth24plus-stencil8','depth32float','depth32float-stencil8']
}
    
    
const particleColor = v4(.3,.3,.8,0)

const QUINN = { url:'quinn.obj', texUrl:'quinn.png', offset:v3(0,0,0), sample:true,
                scale:v3(2, 2, 2), fext:v3(0, 0, 0), color:v4(1,.9,.9,1) }

const HAND = { url:'hand.obj', texUrl:'hand.png', color:v4(.9,.9,.9,1), 
               offset:v3(.7,-.5,.6), scale:v3(1,1,1),
               sample:true, fext:v3(-.5, .1, .5) }

const TORUS = { url:'torus.obj', offset:v3(0,0,2), scale:v3(1),
                color:v4(.7,.2,.1,.89), particleColor:v4(.7,.7,.1,0), sample:true, fext:v3(0,0,-3) }

const GROUND = { url:'ground.obj', texUrl:'marble.png', fext:v3(0),
                 sample:true, lock:()=>0.99 }

const CUBE = { url:'cube.obj', sample:true, color:v4(.3,.3,.8,.7), fext:v3(0), scale:v3(1) }

const TRI = { url:'tri.obj', sample:true, color:v4(0.1,0.4,0.6,0.7), fext:v3(0) }

const WALL = { url:'wall2.obj', fext:v3(0), sample:true, color:v4(0.2,0.2,0.2,.4), lock:()=>0.99 }

const TETRA = { url:'tetra.obj', fext:v3(0,0,0), color:v4(1,1,1,1), fshape:0,
                sample:true, lock:()=>0.9 }

const KNOT = { url:'knot.obj', color:v4(.6,.3,.3,.8), fext:v3(0), offset:v3(0,0,1), sample:true, scale:v3(1)}

const HELPER = { url: 'helper.obj', scale: v3(2,2,2), sample: false, fext: v3(0) }


const lightUpd = (orig,cur,t,off) => {
    let R = 2.5, r = 0.3, l = 0.6, k = .1
    cur.pos.x = R*((1-k)*cos(t+off) - l*k*cos(t*(1-k)/k+off))
    cur.pos.y = R*((1-k)*sin(t+off) - l*k*sin(t*(1-k)/k+off))
}
const LIGHTS = [
    { power: 2, color: v3(1,.85,.6), pos:v3(0,0,2.3), update(l, t) { lightUpd(this,l,t,PI/4) }},
    { power: 2, color: v3(1,.85,.6), pos:v3(0,0,2.3), update(l, t) { lightUpd(this,l,t,3*PI/4) }},
    { power: 2, color: v3(1,.85,.6), pos:v3(0,0,2.3), update(l, t) { lightUpd(this,l,t,5*PI/4) }},
    { power: 2, color: v3(1,.85,.6), pos:v3(0,0,2.3), update(l, t) { lightUpd(this,l,t,7*PI/4) }},
]
      
const MESHES = [
    GROUND,
    TORUS

    //{ url:objfile, color:v4(1,0,0,1), fext:v3(0), offset:v3(-2.5,3,0) },
    //{ url:objfile, color:v4(1,.5,0,.5), fext:v3(0), offset:v3(-1.5,3.5,0) },
    //{ url:objfile, color:v4(1,1,0,1), fext:v3(0), offset:v3(-.5,4,0) },
    //{ url:objfile, color:v4(0,1,0,.5), fext:v3(0), offset:v3(.5,4.5,0) },
    //{ url:objfile, color:v4(0,0,1,.5), fext:v3(0), offset:v3(1.5,5,0) },
    //{ url:objfile, color:v4(1,0,1,1), fext:v3(0), offset:v3(2.5,5.5,0) },
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
        ['camPos', V3],
        ['fcol', f32],
        ['fshape', f32],
        ['friction', f32],
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
        await gpu.init(width,height,ctx,renderPref)

        this.camPos = CAM_POS
        this.camLR = 0
        this.camUD = CAM_UD
        
        const objs = []
        for (const opt of MESHES) {
            const obj = await this.loadObj(opt)
            if (obj.sample) {
                const grid = new VoxelGrid(obj.verts, obj.faces, D)
                grid.voxelize()
                obj.virts = grid.samples
                obj.virtgrads = grid.gradients
                obj.vertToVirt = grid.vertidxs
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
                if (obj.virtgrads) p.grad0 = p.grad = obj.virtgrads[i]
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
        this.gpu.configure(this.ctx,width,height,renderPref)
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
        let lastStamp = null
        
        while (true) {
            await render.step()
            const stamp = performance.now()/1000
            if (lastStamp != null) {
                const tstep = (stamp - lastStamp) / FRAMERATIO
                for (let i of range(FRAMERATIO))
                    await compute.step(tstep)
            }
            await gpu.dev.queue.onSubmittedWorkDone()
            lastStamp = stamp
        }
        
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
        const wgsl = (await fetchtext('./compute.wgsl')).interp({threads, MAXNN, D})
        
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
        const collisions = gpu.computePipe({ shader, entryPoint:'collisions', binds: ['particles','params'] })
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
            gpu.computePass({ pipe:collisions, dispatch:pd, binds:{ particles:bufs.particles, params:bufs.params } }),
            gpu.computePass({ pipe:collisions, dispatch:pd, binds:{ particles:bufs.particles, params:bufs.params } }),
            gpu.computePass({ pipe:collisions, dispatch:pd, binds:{ particles:bufs.particles, params:bufs.params } }),
            gpu.timestamp('stabilize collisions'),
            gpu.computePass({ pipe:project, dispatch:pd, binds:{ particles:bufs.particles, meshes:bufs.meshes, params:bufs.params } }),
            gpu.timestamp('project'),

        )
        
        this.batch = gpu.encode(cmds)
        this.fwdstep = false
        this.tstart = this.tsim = this.tlast = clock()
        this.frames = 0
        this.tsteps = []
        this.profiles = []

    }

    async stats() {
        let ret = { kind:'compute', tstart:this.tstart, tlast:this.tlast, frames:this.frames, tsim:this.tsim }
        if (this.batch) {
            let data = new BigInt64Array(await this.sim.gpu.read(this.batch.stampBuf))
            let labels = this.batch.stampLabels
            ret.profile = Array.from(range(1,labels.length)).map(i => [labels[i], data[i] - data[i-1]])
        }
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

    async step(tstep) {
        const { gpu, particles, bufs, meshes, possessed, params, camPos } = this.sim
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
            params.camPos = camPos
            gpu.write({ buf:bufs.params, data: params })
            this.batch.execute()
        }
        this.tsim += tsmooth
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

        gpu.configure(ctx, width, height, renderPref)
        
        camera.r = D/2
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

        const partPipe = gpu.renderPipe({
            shader, vert:'vert_part', frag:'frag_part', binds: ['meshes', 'camera', 'lights'], topology: 'triangle-strip',
            vertBufs: [{ buf:bufs.particles, arrayStride:Particles.stride, stepMode: 'instance',
                         attributes: [{ shaderLocation:0, offset:Particle.si.off, format:'float32x3' },
                                      { shaderLocation:1, offset:Particle.mesh.off, format:'uint32' }]}] })
        const lightPipe = gpu.renderPipe({ shader, vert:'vert_light', frag:'frag_light', binds: ['camera','lights'],
                                           topology: 'triangle-strip', atc: false, color_src: 'one', color_dst:'one',
                                           alpha_src:'one', alpha_dst:'one' })
        const surfPipe = gpu.renderPipe({
            shader, vert:'vert_surf', frag:'frag_surf',
            binds: ['meshes', 'camera', 'lights', 'tex', 'samp'], 
            vertBufs: [{ buf:bufs.tris, arrayStride:TriVert.size,
                         attributes: [{ shaderLocation:0, offset:TriVert.pos.off, format:'float32x3' },
                                      { shaderLocation:1, offset:TriVert.norm.off, format:'float32x3' },
                                      { shaderLocation:2, offset:TriVert.mesh.off, format:'uint32' },
                                      { shaderLocation:3, offset:TriVert.uv.off, format:'float32x2' }]}]
        })
        const axisPipe = gpu.renderPipe({ shader, vert:'vert_axis', frag:'frag_axis', binds:['camera'], topology:'line-list' })
        const normPipe = gpu.renderPipe({
            shader, vert:'vert_norm', frag:'frag_norm', binds: ['camera'], topology: 'line-list',
            vertBufs: [{ buf:bufs.tris, arrayStride:TriVert.size, stepMode: 'instance',
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

        
        const draws = [
            gpu.draw({ pipe:surfPipe, dispatch:ntris*3, binds:{ meshes: bufs.meshes, camera:bufs.camera, lights:bufs.lights, tex, samp }}),
            gpu.draw({ pipe:partPipe, dispatch:[8, particles.length], binds:{ meshes: bufs.meshes, camera:bufs.camera, lights:bufs.lights }}),
        ]

        if (renderPref.axes) draws.push(gpu.draw({ pipe:axisPipe, dispatch:[2, 3], binds:{ camera:bufs.camera }}))
        if (renderPref.normals) draws.push(gpu.draw({ pipe:normPipe, dispatch:[2, ntris*3], binds:{ camera:bufs.camera }}))
        if (renderPref.grads) draws.push(gpu.draw({ pipe:gradPipe, dispatch:[2, particles.length], binds:{ camera:bufs.camera }}))
        draws.push(gpu.draw({ pipe:lightPipe, dispatch:[14, lights.length], binds: { camera:bufs.camera, lights:bufs.lights }}))
        cmds.push(gpu.renderPass(draws))
        cmds.push(gpu.timestamp('draws'))

        this.batch = gpu.encode(cmds)
        this.profiles = []
        this.tstart = this.tlast = clock()
        this.frames = 0
    }

    async stats() {
        const ret = { kind:'render', tstart:this.tstart, tlast:this.tlast, frames:this.frames }
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
        
        camera.pos = camPos
        camera.ratio = width/height
        camera.projection = M4.perspective(FOV, camera.ratio, .01, 200)
        camera.forward = this.sim.camFwd()
        camera.modelview = M4.look(camPos, camera.forward, UP)
        gpu.write({ buf:bufs.camera, data: camera})
        
        const t = performance.now()/1000
        for (const [i,l] of enumerate(LIGHTS))
            l.update(lights[i], t)        
        gpu.write({ buf:bufs.lights, data: lights})

        params.camPos = camPos
        gpu.write({ buf:bufs.params, data: params })
        
        this.batch.execute()
        this.frames++
        this.tlast = clock()



        


        //let buf = await gpu.read(bufs.tris)
        //globalThis.tris = new Triangles(buf)
        //buf = await gpu.read(bufs.particles)
        //globalThis.particles = new Particles(buf)
        //buf = await gpu.read(bufs.vertices)
        //globalThis.verts = new Vertices(buf)
        //buf = await gpu.read(bufs.meshes)
        //globalThis.meshes = new Meshes(buf)
            
        
    }

}














