import * as util from './utils.js'
import * as gpu from './gpu.js'
import * as geo from './geometry.js'
Object.assign(globalThis, util, gpu, geo)

const MAXNN = 18
const MAXEDGES = 18
const TETLIM = sqrt(2)

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
phys.addNum('gravity', 9.8, -5, 20, 0.1)
phys.addNum('shape_stiff', 0, 0, 1, 0.005)
phys.addNum('vol_stiff', .5, 0, 1, 0.005)
phys.addNum('friction', 0.5, 0, 1, .01)
phys.addNum('airdamp', 0.5, 0, 1, .001)
phys.addNum('collidamp', .1, 0, 1, .001)

export const render = new Preferences('render')
render.addBool('particles', true)
render.addBool('fluid', true)
render.addBool('ground', true)
render.addBool('normals', false)
render.addBool('edges', false)
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
        ['c0', V3],
        ['ci', V3],
        ['pi', u32],
        ['pf', u32],
        ['vi', u32],
        ['vf', u32],
        ['quat', V4],
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
        ['padding', GPU.array({ type:u32, length:25 })]
    ]
})

export const Vertex = GPU.struct({
    name:'Vertex',
    fields:[
        ['pos', V3],
        ['mesh', u32],
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
        ['delta_pos', V3],
        ['nedges', u32],
        ['norm0', V3],
        ['sdf', f32],
        ['norm', V3],
        ['fixed', u32],
        ['quat', V4],
        ['nn', GPU.array({ type:u32, length:MAXNN })],
        ['edges', GPU.array({ type:u32, length:MAXEDGES })],
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

export const pointCloud = async (meshId, verts, transaction) => {
    let particles = Particles.alloc(verts.size)
    for (const [pidx,[vertId, vert]] of enumerate(verts))
        particles[pidx].pos = vert.pos
    let empty = new ArrayBuffer(0)
    let cache = { meshId, particles: particles.buffer, tets:empty, verts:empty, faces:empty, edges:empty, tetGroups:[] }
    transaction.objectStore('cache').put(cache)
}

export const sampleMesh = async (meshId, D, transaction) => {
    let [mesh,verts,faceData] = await Promise.all([
        transaction.query('meshes', { key:meshId }),
        transaction.query('verts', { index:'meshId', key:meshId }),
        transaction.query('faces', { index:'meshId', key:meshId }),
    ])
    mesh = mesh.get(meshId)
    for (let [id,vert] of verts) vert.pos = v3(...vert.pos).mul(mesh.scale).add(mesh.offset)
    if (faceData.size == 0)
        return pointCloud(meshId, verts, transaction)

    let tree = new BVHTree([...faceData].map(([,face]) => [0,1,2].map(i => verts.get(face['vertId'+i]).pos)))
    let bmin = v3(Infinity), bmax = v3(-Infinity)
    for (const [vertId, vert] of verts) {
        vert.loop = new Map()
        bmin = bmin.min(vert.pos)
        bmax = bmax.max(vert.pos)
    }
    let bounds = bmax.sub(bmin)
    dbg({bounds})
    let dims = [...bounds.divc(D)].map(roundEps).map(ceil).map(d=>max(1,d))
    dbg({dims})
    dims = dims.map(d => d + (d%2 == 0 ? 1 : (d > 1 ? 2 : 1)))
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
        let pos = v3(D*x,D*y,D*z).add(offset)
        let query = tree.signedDist({ p:pos })
        if (query.sdf <= D*TETLIM) {
            let p = Particle.alloc()
            p.pos = pos
            p.norm0 = p.norm = query.n
            p.sdf = query.sdf
            hpmap.set(h, p)
            if (mesh.fluid && (x+y+z) % 2 == 0 ) particles.push(p)
        }
    }

    const keepTet = (tet) => {
        for (let p of tet)
            if (tree.signedDist({p}).sdf <= 0)
                return true
        for (let i of range(4)) {
            let faceCenter = v3(0)
            for (let j of range(4))
                faceCenter = faceCenter.add(tet[j].mulc(Number(i != j)))
            faceCenter = faceCenter.divc(3)
            if (tree.signedDist({p:faceCenter}).sdf <= 0)
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
                if (!keepTet(hps.map(([h,p]) => p.pos))) continue
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
        particles: Particles.of(particles),
        verts: Vertices.alloc(verts.size),
        faces: Triangles.alloc(faceData.size)
    }

    let centers = []
    let tmats = []
    for (let [tidx,pids] of enumerate(tets)) {
        cache.tets[tidx] = Tet.of(pids)
        let ps = pids.map(pid => cache.particles[pid].pos)
        centers.push(ps[0].add(ps[1]).add(ps[2]).add(ps[3]).divc(4))
        let tmat =  m3([ps[1].sub(ps[0]), ps[2].sub(ps[0]), ps[3].sub(ps[0])])
        let vol = tmat.invert() / 6
        if (vol < 0) throw new Error('got zero or negative volume')
        let pm = vol / 4
        for (let pid of pids)
            cache.particles[pid].w += pm
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
            v.bary = v.pos.sub(cache.particles[tets[v.tet][0]].pos).mulm(tmats[tetbest.t])
        } else v.tet = -1
    }

    for (let [fidx,face] of enumerate(faceData.values()))
        cache.faces[fidx] = Triangle.of(...[0,1,2].map(i => TriVert.of(
            v3(0), vertIdMap[face['vertId'+i]], v3(0), -1, v2(...face.uv[i]))))

    let edges = new Set()
    for (let [a,b,c,d] of cache.tets)
        for (let [pid1,pid2] of [[a,b],[a,c],[a,d],[b,c],[b,d],[c,d]])
            edges.add(hashPair(pid1,pid2))
    edges = [...edges].map(hash => unhashPair(hash))

    for (let [pid1,pid2] of edges) {
        cache.particles[pid1].edges[cache.particles[pid1].nedges++] = pid2
        cache.particles[pid2].edges[cache.particles[pid2].nedges++] = pid1
    }

    cache.edges = u32array.of(edges.flat())

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



export async function Sim(width, height, ctx) {

    const gpu = new GPU()
    await gpu.init(width,height,ctx)

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
        mdata.parts = new Particles(cache.particles)
        mdata.tets = new Tets(cache.tets)
        mdata.mass = new Float32Array(cache.mass)
        mdata.verts = new Vertices(cache.verts)
        mdata.faces = new Triangles(cache.faces)
        mdata.edges = new u32array(cache.edges)
        mdata.tetGroups = cache.tetGroups
    }

    let meshes = Meshes.alloc(meshData.length)
    let particles = Particles.alloc(meshData.map(([,mdata]) => mdata.parts.length).sum())
    let verts = Vertices.alloc(meshData.map(([,mdata]) => mdata.verts.length).sum())
    let tris = Triangles.alloc(meshData.map(([,mdata]) => mdata.faces.length).sum())
    let tets = Tets.alloc(meshData.map(([,mdata]) => mdata.tets.length).sum())
    let edges = u32array.alloc(meshData.map(([,mdata]) => mdata.edges.length).sum())
    let tetGroups = []
    let haveFluids = false

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
        mesh.tex = bitmapIds[mdata.bitmapId]
        mesh.quat = v4(0,0,0,1)
        mesh.pi = nparticles
        mesh.pf = nparticles + mdata.parts.length
        mesh.vi = nverts
        mesh.vf = nverts + mdata.verts.length
        mesh.fluid = mdata['fluid']
        haveFluids = haveFluids || mesh.fluid == 1
        let ti = ntets
        let wavg = 0
        let ws = new Set()
        for (let pidx of range(mdata.parts.length))  {
            let p = mdata.parts[pidx]
            p.prev_pos = p.rest_pos = p.pos
            mesh.c0 = mesh.c0.add(p.pos)
            p.mesh = midx
            p.quat = v4(0,0,0,1);
            p.w = 1.0 / p.w * mdata.invmass
            p.fixed = mdata.fixed
            ws.add(p.w)
            for (let i of range(p.nedges))
                p.edges[i] += mesh.pi
            particles[nparticles++] = p
        }
        mesh.c0 = mesh.ci = mesh.c0.divc(mesh.pf - mesh.pi)

        dbg({ws:[...ws].sort((a,b)=>a-b)})

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
        buf:gpu.buf({ label:`tetgroup${i}`, data:u32array.of(group), usage:'STORAGE' }), data:group
    }))

    const uniforms = Uniforms.alloc()
    uniforms.selection = uniforms.grabbing = -1
    const lights = GPU.array({ type:Light, length:LIGHTS.length }).alloc(LIGHTS.length)
    for (const [i,l] of enumerate(LIGHTS)) {
        lights[i].color = l.color
        lights[i].pos = l.pos
        lights[i].power = l.power
    }

    dbg({ nparts:particles.length, nverts:verts.length, nmeshes:meshes.length, ntets:tets.length, ntris:tris.length })
    const threads = gpu.threads

    const bufs = Object.fromEntries([
        gpu.buf({ label:'particles', data:particles, usage:'STORAGE|COPY_SRC|COPY_DST|VERTEX' }),
        gpu.buf({ label:'tets', data:tets, usage:'STORAGE|COPY_DST|COPY_SRC' }),
        gpu.buf({ label:'edges', data:edges, usage:'STORAGE|COPY_DST|COPY_SRC|INDEX' }),
        gpu.buf({ label:'vertices', data:verts, usage:'STORAGE|VERTEX|COPY_SRC' }),
        gpu.buf({ label:'meshes', data:meshes, usage:'STORAGE|COPY_DST|COPY_SRC' }),
        gpu.buf({ label:'uniforms', data:uniforms, usage:'UNIFORM|COPY_DST' }),
        gpu.buf({ label:'tris', data:tris, usage:'STORAGE|VERTEX|COPY_SRC|COPY_DST' }),
        gpu.buf({ label:'cnts', type:GPU.array({ type:i32, length:threads**3 }), usage:'STORAGE|COPY_DST|COPY_SRC' }),
        gpu.buf({ label:'work1', type:GPU.array({ type:i32, length:threads**2 }), usage:'STORAGE' }),
        gpu.buf({ label:'work2', type:GPU.array({ type:i32, length:threads }), usage:'STORAGE' }),
        gpu.buf({ label:'sorted', type:GPU.array({ type:u32, length:particles.length }), usage:'STORAGE' }),
        gpu.buf({ label:'lights', data:lights, usage:'UNIFORM|FRAGMENT|COPY_DST' }),
        gpu.buf({ label:'fluidtex', type:GPU.array({ type:i32, length:width*height }), usage:'STORAGE|COPY_DST' }),
        gpu.buf({ label:'debug', type:GPU.array({ type: f32, length:4096 }), usage:'STORAGE|COPY_SRC|COPY_DST' }),
    ].map(buf => [buf.label, buf]))


    const pd = ceil(particles.length/gpu.threads)

    const syncUniforms = () => {
        uniforms.r = phys.r
        uniforms.cam_pos = render.cam_pos
        uniforms.width = width
        uniforms.height = height
        uniforms.cam_fwd = v3(sin(render.cam_lr) * cos(render.cam_ud),
                              cos(render.cam_lr) * cos(render.cam_ud),
                              sin(render.cam_ud)).normalized()
        uniforms.proj = M4.perspective(render.fov, width/height, .01, 10000)
        uniforms.view = M4.look(render.cam_pos, uniforms.cam_fwd, v3(0,0,1))
        uniforms.mvp = uniforms.view.mul(uniforms.proj)

        phys.keys.filter(k => phys.type[k] == 'num').forEach(k => {
            uniforms[k] = phys[k]
        })
        uniforms.ground = render.ground ? 1 : 0
        gpu.write(bufs.uniforms, uniforms)
    }

    async function Computer() {

        let fwdstep = false
        let tstart = clock(), tlast = clock()
        let frames = 0, steps = 0

        if (particles.length == 0) return


        const threads = gpu.threads
        const wgsl = (await fetchtext('./compute.wgsl')).interp({ threads, MAXNN })

        const shader = await gpu.shader({
            compute:true, wgsl:wgsl, defs:[Particle, Mesh, Uniforms],
            storage:{
                particles:Particles, meshes:Meshes, sorted:u32array, centroidwork:v3array, edges:Edges, debug:f32array,
                tetgroup:u32array, cnts:i32array, cnts_atomic:iatomicarray, work:i32array, shapework:m3Array, tets:Tets
            },
            uniform:{ uniforms:Uniforms }
        })

        const cmds = [gpu.timestamp('')]

        const predict = gpu.computePipe({ shader, entryPoint: 'predict', binds: ['particles', 'meshes', 'uniforms'] })
        const cntsort_cnt = gpu.computePipe({ shader, entryPoint: 'cntsort_cnt',
            binds: ['particles', 'cnts_atomic', 'meshes', 'uniforms'] })
        const prefsum_down = gpu.computePipe({ shader, entryPoint: 'prefsum_down', binds: ['cnts', 'work'] })
        const prefsum_up = gpu.computePipe({ shader, entryPoint: 'prefsum_up', binds: ['cnts', 'work'] })
        const cntsort_sort = gpu.computePipe({ shader, entryPoint: 'cntsort_sort',
            binds: ['particles', 'meshes', 'cnts_atomic', 'sorted'] })
        const find_collisions = gpu.computePipe({ shader, entryPoint: 'find_collisions',
            binds: ['particles', 'meshes', 'cnts', 'sorted', 'uniforms'] })
        const collide = gpu.computePipe({ shader, entryPoint: 'collide', binds: ['particles', 'meshes', 'uniforms', 'debug'] })
        const update_vels = gpu.computePipe({ shader, entryPoint:'update_vel', binds:['particles', 'meshes', 'uniforms', 'debug'] })
        const centroid_prep = gpu.computePipe({ shader, entryPoint: 'centroid_prep', binds: ['meshes', 'particles', 'centroidwork']})
        const get_centroid = gpu.computePipe({ shader, entryPoint: 'get_centroid', binds: ['meshes', 'centroidwork'] })
        const rotate_prep = gpu.computePipe({ shader, entryPoint: 'rotate_prep', binds: ['meshes', 'particles', 'shapework'] })
        const get_rotate = gpu.computePipe({ shader, entryPoint: 'get_rotate', binds: ['meshes', 'shapework'] })
        const normalmatch = gpu.computePipe({ shader, entryPoint:'normalmatch', binds:['particles', 'meshes', 'uniforms', 'debug'] })
        const neohookean = gpu.computePipe({ shader, entryPoint:'neohookean',
            binds:['particles', 'meshes', 'tets', 'uniforms','debug','tetgroup'] })
        const shapematch = gpu.computePipe({ shader, entryPoint:'shapematch', binds:['particles', 'meshes', 'uniforms', 'debug'] })

        cmds.push(
            gpu.computePass({ pipe:predict, dispatch:pd,
                binds:{ particles:bufs.particles, meshes:bufs.meshes, uniforms:bufs.uniforms } }),
            gpu.timestamp('predict')
        )

        cmds.push(
            gpu.clearBuffer(bufs.cnts, 0, bufs.cnts.size),
            gpu.computePass({ pipe:cntsort_cnt, dispatch:pd,
                binds:{ particles:bufs.particles, cnts_atomic:bufs.cnts, meshes:bufs.meshes, uniforms:bufs.uniforms }}),
            gpu.timestamp('cntsort_cnt'),
            gpu.computePass({ pipe:prefsum_down, dispatch:threads ** 2, binds:{ cnts:bufs.cnts, work:bufs.work1 }}),
            gpu.computePass({ pipe:prefsum_down, dispatch:threads, binds:{ cnts:bufs.work1, work:bufs.work2 }}),
            gpu.computePass({ pipe:prefsum_down, dispatch:1, binds:{ cnts:bufs.work2, work:bufs.work2 }}),
            gpu.computePass({ pipe:prefsum_up, dispatch:threads - 1, binds:{ cnts:bufs.work1, work:bufs.work2 }}),
            gpu.computePass({ pipe:prefsum_up, dispatch:threads**2 - 1, binds:{ cnts:bufs.cnts, work:bufs.work1 }}),
            gpu.timestamp('prefsum'),
            gpu.computePass({ pipe:cntsort_sort, dispatch:pd,
                binds:{ particles:bufs.particles, meshes:bufs.meshes, cnts_atomic:bufs.cnts, sorted:bufs.sorted }}),
            gpu.timestamp('cntsort_sort'),
            gpu.computePass({ pipe:find_collisions, dispatch:pd,
                binds:{ particles:bufs.particles, meshes:bufs.meshes, cnts:bufs.cnts, sorted:bufs.sorted, uniforms:bufs.uniforms }}),
            gpu.timestamp('find collisions')
        )


        cmds.push(
            gpu.computePass({ pipe:collide, dispatch:pd,
                binds:{ particles:bufs.particles, meshes:bufs.meshes, uniforms:bufs.uniforms, debug:bufs.debug } }),
            gpu.timestamp('collide'),
        )

        for (const [i, m] of enumerate(meshes)) {
            let n = m.pf - m.pi
            if (n <= 0) continue
            if (m.flags == 1) continue
            const centroidWork = gpu.buf({ label: `centroidwork${i}`, type: v3array, size: v3array.stride * n, usage: 'STORAGE' })
            const shapeWork = gpu.buf({ label: `shapework${i}`, type: m3Array, size: m3Array.stride * n, usage: 'STORAGE' })
            let dp1 = ceil(n / threads), dp2 = ceil(dp1 / threads)
            const meshBind = { meshes: gpu.offset(bufs.meshes, Meshes.stride * i) }
            cmds.push(gpu.computePass({ pipe: centroid_prep, dispatch: dp1,
                binds: { ...meshBind, particles: bufs.particles, centroidwork: centroidWork } }))
            cmds.push(gpu.computePass({ pipe: get_centroid, dispatch: dp1, binds: { ...meshBind, centroidwork: centroidWork } }))
            if (dp1 > 1) {
                cmds.push(gpu.computePass({ pipe: get_centroid, dispatch: dp2, binds: { ...meshBind, centroidwork: centroidWork } }))
                if (dp2 > 1)
                    cmds.push(gpu.computePass({ pipe: get_centroid, dispatch: 1, binds: { ...meshBind, centroidwork: centroidWork } }))
            }
            if (n <= 1) continue
            cmds.push(gpu.computePass({ pipe: rotate_prep, dispatch: dp1,
                binds: { ...meshBind, particles: bufs.particles, shapework: shapeWork } }))
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
                pipe:normalmatch, dispatch:pd,
                binds:{ particles:bufs.particles, meshes:bufs.meshes, uniforms:bufs.uniforms, debug:bufs.debug },
            }),
            gpu.timestamp('normal match')
        )

        cmds.push(...tetGroups.map(group => gpu.computePass({
            pipe: neohookean, dispatch:ceil(group.data.length/gpu.threads),
            binds:{ particles:bufs.particles, meshes: bufs.meshes, tets: bufs.tets,
                    uniforms: bufs.uniforms, debug:bufs.debug, tetgroup:group.buf }
        })))
        cmds.push(gpu.timestamp('neohookean'))

        cmds.push(
            gpu.computePass({
                pipe: shapematch, dispatch:pd,
                binds:{ particles:bufs.particles, meshes:bufs.meshes, uniforms:bufs.uniforms, debug:bufs.debug },
            }),
            gpu.timestamp('shape match')
        )

        cmds.push(
            gpu.computePass({
                pipe:update_vels, dispatch:pd,
                binds:{ particles:bufs.particles, meshes:bufs.meshes, uniforms:bufs.uniforms, debug:bufs.debug }
            }),
            gpu.timestamp('update vel')
        )

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

                   /* const reads = [
                        'particles',
                        'debug'
                    ].filter(b=>bufs[b])

                    const data = await Promise.all(reads.map(b => gpu.read(bufs[b])))
                    for (let i of range(reads.length))
                        globalThis[reads[i]] = new bufs[reads[i]].type(data[i])
                    window.debug = [...debug]*/

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

            const vd = ceil(verts.length/gpu.threads)
            const td = ceil(tris.length/gpu.threads)


            let wgsl = (await fetchtext('./prerender.wgsl')).interp({threads: gpu.threads})
            const preShader = await gpu.shader({
                wgsl, compute: true, defs:[ Vertex, Mesh, Particle, Uniforms, TriVert, Triangle ],
                storage:{  particles:Particles, meshes:Meshes, vertices:Vertices, tris:Triangles, tets:Tets, /*fluidtex:iatomicarray, debug:f32array*/ },
                uniform: { uniforms:Uniforms }
            })

            wgsl = await fetchtext('./render.wgsl')
            wgsl = wgsl.interp({numLights: lights.length })
            const shader = await gpu.shader({ wgsl, defs:[Vertex, Particle, Mesh, Uniforms, Light],
                                            storage:{ particles:Particles, meshes:Meshes, vertices:Vertices, /*fluidtex:i32array*/ },
                                            uniform:{ uniforms:Uniforms, lights:lights.constructor },
                                            textures:{ tex:{ name:'texture_2d_array<f32>' } },
                                            samplers:{ samp:{ name:'sampler' } } })

            let tex = gpu.texture(bitmaps)
            let samp = gpu.sampler()

            const cmds = []
            cmds.push(gpu.timestamp(''))
            if (vd > 0 && particles.length > 0) {
                if (bufs.tets.data.length > 0) {
                    const vertpos = gpu.computePipe({
                        shader: preShader, entryPoint:'vertpos', binds: ['vertices','particles','meshes','tets','debug'] })
                    cmds.push(gpu.computePass({ pipe:vertpos, dispatch:vd,
                        binds:{ vertices:bufs.vertices, particles:bufs.particles, meshes:bufs.meshes, tets:bufs.tets, debug:f32array} }))
                    cmds.push(gpu.timestamp('vertexpositions'))
                }
                const normals = gpu.computePipe({ shader: preShader, entryPoint:'normals', binds: ['vertices','particles'] })
                cmds.push(gpu.computePass({ pipe:normals, dispatch:vd,
                    binds:{ vertices:bufs.vertices, particles:bufs.particles } }))
                cmds.push(gpu.timestamp('vertexnormals'))
            }
            if (td > 0) {
                const updTris = gpu.computePipe({ shader: preShader, entryPoint:'update_tris', binds: ['tris','vertices'] })
                cmds.push(gpu.computePass({ pipe:updTris, dispatch:td, binds:{ vertices:bufs.vertices, tris:bufs.tris } }))
                cmds.push(gpu.timestamp('vertexnormals'))
            }

            /*if (haveFluids && render.fluid) {
                const prefluid = gpu.computePipe({ shader: preShader, entryPoint:'fluids', binds: ['particles','meshes','fluidtex','uniforms','debug'] })
                cmds.push(gpu.clearBuffer(bufs.fluidtex))
                cmds.push(gpu.computePass({ pipe:prefluid, dispatch:pd, binds:{ particles:bufs.particles, meshes:bufs.meshes, fluidtex:bufs.fluidtex, debug:bufs.debug, uniforms:bufs.uniforms } }))
                cmds.push(gpu.timestamp('prefluids'))
            }*/


            const draws = []
            if (render.ground) {
                const gndPipe = gpu.renderPipe({ shader, entry:'ground', cullMode:'none',
                    binds:['uniforms','lights'], topology:'triangle-strip' })
                draws.push(gpu.draw({ pipe:gndPipe, dispatch:8, binds:{ uniforms:bufs.uniforms, lights:bufs.lights }}))
            }

            if (tris.length > 0) {
                const surfPipe = gpu.renderPipe({
                    shader, entry:'surface', binds: ['meshes', 'uniforms', 'lights', 'tex', 'samp'],
                    vertBufs: [{ buf:bufs.tris, arrayStride:TriVert.size,
                                 attributes: [{ shaderLocation:0, offset:TriVert.pos.off, format:'float32x3' },
                                              { shaderLocation:1, offset:TriVert.norm.off, format:'float32x3' },
                                              { shaderLocation:2, offset:TriVert.mesh.off, format:'uint32' },
                                              { shaderLocation:3, offset:TriVert.uv.off, format:'float32x2' }]}]})
                draws.push(gpu.draw({ pipe:surfPipe, dispatch:tris.length*3,
                    binds:{ meshes: bufs.meshes, uniforms:bufs.uniforms, lights:bufs.lights, tex, samp }}))
            }


            if (render.particles && particles.length > 0) {
                const partPipe = gpu.renderPipe({
                    shader, entry:'particle', binds: ['meshes', 'uniforms', 'lights'], topology: 'triangle-list',
                    vertBufs: [{ buf:bufs.particles, arrayStride:Particles.stride, stepMode: 'instance',
                                 attributes: [{ shaderLocation:0, offset:Particle.pos.off, format:'float32x3' },
                                              { shaderLocation:1, offset:Particle.mesh.off, format:'uint32' }]}] })
                draws.push(gpu.draw({ pipe:partPipe, dispatch:[3, particles.length],
                    binds:{ meshes: bufs.meshes, uniforms:bufs.uniforms, lights:bufs.lights }}))
            }

            if (render.normals) {
                const normPipe = gpu.renderPipe({
                    shader, entry:'vnormals', binds: ['uniforms'], topology: 'line-list',
                    vertBufs: [{ buf:bufs.tris, arrayStride:TriVert.size, stepMode: 'instance',
                    attributes: [{ shaderLocation:0, offset:TriVert.pos.off, format:'float32x3' },
                                 { shaderLocation:1, offset:TriVert.norm.off, format:'float32x3' }]}]})
                draws.push(gpu.draw({ pipe:normPipe, dispatch:[2, tris.length*3], binds:{ uniforms:bufs.uniforms }}))

                if (render.particles && particles.length > 0) {
                    const gradPipe = gpu.renderPipe({
                        shader, entry:'pnormals', binds: ['uniforms'], topology: 'line-list',
                        vertBufs: [{ buf:bufs.particles, arrayStride:Particles.stride, stepMode: 'instance',
                                     attributes: [{ shaderLocation:0, offset:Particle.pos.off, format:'float32x3' },
                                                  { shaderLocation:1, offset:Particle.norm.off, format:'float32x3' }]}]})
                    draws.push(gpu.draw({ pipe:gradPipe, dispatch:[2, particles.length], binds:{ uniforms:bufs.uniforms }}))
                }

            }

            if (render.edges && edges.length > 0) {
                const edgePipe = gpu.renderPipe({
                    shader, entry:'edges', binds:['uniforms'], topology:'line-list',
                    indexBuf: { buf: bufs.edges, indexFormat: 'uint32' },
                    vertBufs: [{
                        buf:bufs.particles, arrayStride:Particles.stride, stepMode:'vertex',
                        attributes: [{ shaderLocation:0, offset:Particle.pos.off, format:'float32x3'}]
                    }]
                })
                draws.push(gpu.drawIndexed({ pipe:edgePipe, dispatch:edges.length, binds:{ uniforms:bufs.uniforms }}))
            }



            if (haveFluids && render.fluid) {

                //const fluidQuadPipe = gpu.renderPipe({ shader, entry:'fluidquad', topology:'triangle-strip',  binds:['uniforms','lights','fluidtex'] })
                //draws.push(gpu.draw({ pipe:fluidQuadPipe, dispatch: [4,1], binds: { uniforms:bufs.uniforms, lights:bufs.lights, fluidtex:bufs.fluidtex }}))

                const fluidPipe = gpu.renderPipe({
                    shader, entry:'fluid', binds: ['meshes', 'uniforms', 'lights', 'particles'], topology: 'triangle-list',
                    atc:false, depthWriteEnabled:false, cullMode:'none',
                    vertBufs: [{ buf:bufs.particles, arrayStride:Particles.stride, stepMode: 'instance',
                                 attributes: [{ shaderLocation:0, offset:Particle.pos.off, format:'float32x3' }]}] })
                draws.push(gpu.draw({
                    pipe:fluidPipe, dispatch:[3, particles.length],
                    binds:{ meshes: bufs.meshes, uniforms:bufs.uniforms, lights:bufs.lights, particles:bufs.particles }}))
            }

            const lightPipe = gpu.renderPipe({ shader, entry:'lights', binds: ['uniforms','lights'], topology: 'triangle-list',
                atc:false, depthWriteEnabled:false })
            draws.push(gpu.draw({ pipe:lightPipe, dispatch:[3, lights.length],
                binds: { uniforms:bufs.uniforms, lights:bufs.lights }}))


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
                gpu.write(bufs.lights, lights)
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
        bufs.fluidtex = gpu.buf({ label:'fluidtex', type:GPU.array({ type:i32, length:width*height }), usage:'STORAGE' })
    }


     async function run () {
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
        let rsq = (phys.r)**2
        let particles = new Particles(await gpu.read(bufs.particles))

        let hitdists = []
        for (const [i, p] of enumerate(particles)) {
            if (meshes[p.mesh].inactive == 1) continue
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
        dbg({pos:p.pos, w:p.w, nedges:p.nedges})
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
        let buf = gpu.chop(gpu.offset(bufs.particles, Particles.stride * pid), Particle.size)
        let p = new Particle(await gpu.read(buf))
        p.fixed = p.fixed == 1 ? 0 : 1
        dbg({fixed:p.fixed})
        gpu.write(buf, p)
    }

    return scope(stmt => eval(stmt))

}






export const loadWavefront = async (name, data, transaction) => {
    let meshes = transaction.objectStore('meshes')
    let meshId = await transaction.wait(meshes.add({
        name, bitmapId:-1, color:[1,1,1,1], offset:[0,0,0], rotation:[0,0,0], gravity:1, invmass:1,
        scale:[1,1,1], 'shape stiff':1, 'vol stiff':1, friction:1, 'collision damp':1, fluid:0, fixed: 0
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






