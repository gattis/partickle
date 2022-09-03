const module = globalThis
const { abs,cos,sin,acos,asin,cbrt,sqrt,pow,PI,random,round,ceil,floor,tan,max,min,log2 } = Math
import './gpu.js'



module.intersectRayTri = (start, dir, a, b, c) => {
    const ab = b.sub(a)
    const ac = c.sub(a)
    const n = ab.cross(ac)
    const ndir = dir.mulc(-1)
    const d = ndir.dot(n)
    const ood = 1/d
    const ap = start.sub(a)
    const t = ap.dot(n) * ood
    if (t < 0) return Infinity
    const e = ndir.cross(ap)
    const v = ac.dot(e) * ood
    if (v < 0 || v > 1) return Infinity
    const w = -ab.dot(e) * ood
    if (w < 0 || v+w > 1) return Infinity
    return t
}

module.intersectRayAABB = (start, dir, lower, upper) => {
    let tx = -1, ty = -1, tz = -1
    let inside = true
    if (start.x < lower.x) {
        inside = false
        if (dir.x != 0) tx = (lower.x-start.x)/dir.x
    } else if (start.x > upper.x) {
        inside = false
        if (dir.x != 0) tx = (upper.x-start.x)/dir.x
    }
    if (start.y < lower.y) {
        inside = false
        if (dir.y != 0) ty = (lower.y-start.y)/dir.y
    } else if (start.y > upper.y) {
        inside = false
        if (dir.y != 0) ty = (upper.y-start.y)/dir.y
    }
    if (start.z < lower.z) {
        inside = false
        if (dir.z != 0) tz = (lower.z-start.z)/dir.z
    } else if (start.z > upper.z) {
        inside = false
        if (dir.z != 0) tz = (upper.z-start.z)/dir.z
    }
    if (inside) return 0
    let tmax = tx
    let taxis = 0
    if (ty > tmax) { tmax = ty; taxis = 1 }
    if (tz > tmax) { tmax = tz; taxis = 2 }
    if (tmax < 0) return Infinity
    let hit = start.add(dir.mulc(tmax))
    if ((hit.x < lower.x || hit.x > upper.x) && taxis != 0) return Infinity
    if ((hit.y < lower.y || hit.y > upper.y) && taxis != 1) return Infinity
    if ((hit.z < lower.z || hit.z > upper.z) && taxis != 2) return Infinity
    return tmax
}

module.traceRay = (verts, tris, start, dir) => {
    let result = { t: Infinity }
    for (const tidx of range(tris.length)) {
        const tri = tris[tidx]
        const a = verts[tri[0].vidx], b = verts[tri[1].vidx], c = verts[tri[2].vidx]
        const t = intersectRayTri(start, dir, a, b, c)
        if (t < result.t) {
            result.t = t
            result.tidx = tidx
        }
    }
    return result
}


module.voxelize = (verts, tris, d) => {

    const r = d/2
    let lower = Vec3.of(Infinity), upper = Vec3.of(-Infinity)
    let centroid = Vec3.of(0,0,0)
    for (let v of verts)
        centroid = centroid.add(v)
    centroid = centroid.divc(verts.length)
    const relverts = []
    for (let v of verts) {
        const relv = v.sub(centroid)
        lower = Vec3.min(lower, relv)
        upper = Vec3.max(upper, relv)
        relverts.push(relv)
    }
    
    const span = upper.sub(lower)
    const nvoxels = span.divc(d*0.9999).floor().max(Vec3.of(1))
    lower = lower.sub(Vec3.of(2*d))
    upper = upper.add(Vec3.of(2*d))
    const start = lower.add(span.sub(nvoxels.mulc(d)).divc(2))
    const dim = nvoxels.addc(4)
    
    const tstart = performance.now()
    const tree = new AABBTree(relverts, tris)

    const voxels = []
    let rayDir = Vec3.of(0,0,1)    
    for (const x of range(dim.x)) {
        for (const y of range(dim.y)) {
            let inside = false
            let rayStart = start.add(Vec3.of(d*x + r, d*y + r, 0))
            let lastTidx = -1
            while (true) {
                const trace = tree.traceRay(rayStart, rayDir)
                if (trace.t == Infinity) break
                const zhit = (trace.t + rayStart.z - start.z) / d
                const z = floor((rayStart.z - start.z + r) / d)
                const zend = min(floor(zhit + 0.5), dim.z - 1)
                if (inside)
                    for (let k = z; k < zend; k++)
                        voxels.push([x,y,k])
                inside = !inside
                if (trace.tidx == lastTidx) throw Error('self intersect')
                lastTidx = trace.tidx
                rayStart = rayStart.add(rayDir.mulc(trace.t + 0.00001 * dim.z * d))                
            }
        }
    }

    const sdf = SDF(voxels, dim)
    const gradients = voxels.map(([x,y,z]) => sdfGrad(sdf, dim, x, y, z).normalized())
    const samples = voxels.map(([x,y,z]) => Vec3.of(start.x+d*x+r+centroid.x, start.y+d*y+r+centroid.y, start.z+d*z+r+centroid.z))
    
    console.log(`voxelize took ${performance.now() - tstart}ms`)
    return { samples, gradients }
}

module.sdfGrad = (sdf, dim, x, y, z) => {
    const dx = sampleGrid(sdf, dim, min(x + 1, dim.x - 1), y, z) - sampleGrid(sdf, dim, max(x - 1, 0), y, z)
    const dy = sampleGrid(sdf, dim, x, min(y + 1, dim.y - 1), z) - sampleGrid(sdf, dim, x, max(y - 1, 0), z)
    const dz = sampleGrid(sdf, dim, x, y, min(z + 1, dim.z - 1)) - sampleGrid(sdf, dim, x, y, max(z - 1, 0))
    const grad = Vec3.of(dx,dy,dz)
    return dim.divc(2).mul(grad)
}



module.sampleGrid = (voxgrid, dim, x, y, z) => {
    return voxgrid[clamp(x, 0, dim.x-1) + clamp(y, 0, dim.y-1)*dim.x + clamp(z, 0, dim.z-1)*dim.x*dim.y]
}

module.edgeDetect = (voxgrid, dim, x, y, z) => {
    const center = sampleGrid(voxgrid, dim, x, y, z)
    let dist = Infinity
    for (const k of [z - 1, z, z + 1])
        for (const j of [y - 1, y, y + 1])
	    for (const i of [x - 1, x, x + 1])
		if (sampleGrid(voxgrid, dim, i, j, k) != center)
		    dist = min(dist, sqrt((x-i)**2 + (y-j)**2 + (z-k)**2) / 2);
    return dist
}

module.SDF = (samples, dim) => {
    const voxgrid = new Uint8Array(dim.x*dim.y*dim.z)
    for (const [x,y,z] of samples)
        voxgrid[x + y*dim.x + z*dim.x*dim.y] = 1
    const queue = new Heap((a,b) => a[3] - b[3])
    const sdf = new Float32Array(dim.x*dim.y*dim.z)
    for (const z of range(dim.z))
        for (const y of range(dim.y))
            for (const x of range(dim.x)) {
                const dist = edgeDetect(voxgrid, dim, x, y, z)
                if (dist != Infinity)
                    queue.push([x,y,z,dist,x,y,z])
                sdf[x + y*dim.x + z*dim.x*dim.y] = Infinity
            }
    while (queue.items.length) {
        const [ci,cj,ck,d,si,sj,sk] = queue.pop()
        const pos = ci + cj*dim.x + ck*dim.x*dim.y
        if (sdf[pos] == Infinity) {
            sdf[pos] = d
            for (const z of [max(0,ck-1), ck, min(ck+1, dim.z-1)])
                for (const y of [max(0,cj-1), cj, min(cj+1, dim.y-1)])
                    for (const x of [max(0,ci-1), ci, min(ci+1, dim.x-1)])
                        if ((ci != x || cj != y || ck != z) && sdf[x + y*dim.x + z*dim.x*dim.y] == Infinity) {
                            const dnext = sqrt((x-si)**2 + (y-sj)**2 + (z-sk)**2) + sdf[si + sj*dim.x + sk*dim.x*dim.y]
                            queue.push([x,y,z,dnext,si,sj,sk])
                        }
        }
    }
    const scale = 1/max(max(dim.x,dim.y),dim.z)
    for (const z of range(dim.z))
        for (const y of range(dim.y))
            for (const x of range(dim.x)) {
                const pos = x + y*dim.x + z*dim.x*dim.y
                sdf[pos] *= scale //(voxgrid[pos] ? -1 : 1) * scale
            }
    return sdf
}


module.AABBTree = class AABBTree {
    constructor(verts, tris) {
        this.verts = verts
        this.tris = tris
        this.nfaces = tris.length
        this.faceUppers = []
        this.faceLowers = []
        this.centroids = [[],[],[]]
        for (const tri of tris) {            
            const a = verts[tri[0].vidx], b = verts[tri[1].vidx], c = verts[tri[2].vidx]
            this.faceLowers.push(a.min(b).min(c))
            this.faceUppers.push(a.max(b).max(c))
            this.centroids[0].push((a.x + b.x + c.x)/3)
            this.centroids[1].push((a.y + b.y + c.y)/3)
            this.centroids[2].push((a.z + b.z + c.z)/3)
        }
        
        this.top = { faces: new Uint32Array(range(this.nfaces)) }

        const recurse = (node) => {
            node.lower = Vec3.of(Infinity)
            node.upper = Vec3.of(-Infinity)            
            for (const face of node.faces) {   
                node.lower = node.lower.min(this.faceLowers[face])
                node.upper = node.upper.max(this.faceUppers[face])
            }
            node.leaf = node.faces.length <= 6
            if (!node.leaf) {
                this.partition(node)
                recurse(node.left)
                recurse(node.right)
            }
        }
  
        recurse(this.top)
        
    }
                     
    partition(node) {
        const faces = node.faces, nfaces = faces.length
        const { verts, tris, faceUppers, faceLowers, centroids } = this
        const best = {axis: 0, idx: 0, cost: Infinity}
        const cmpfn = (axis) => {
            return (a,b) => centroids[axis][a] - centroids[axis][b]
        }
        for (const axis of range(3)) {
            faces.sort(cmpfn(axis))
            const cumLeft = new Float32Array(nfaces), cumRight = new Float32Array(nfaces)
            let leftLower = Vec3.of(Infinity), leftUpper = Vec3.of(-Infinity)
            let rightLower = Vec3.of(Infinity), rightUpper = Vec3.of(-Infinity)
            for (const i of range(nfaces)) {
                let face = faces[i]
                leftLower = leftLower.min(faceLowers[face])
                leftUpper = leftUpper.max(faceUppers[face])
                face = faces[nfaces-i-1]
                rightLower = rightLower.min(faceLowers[face])
                rightUpper = rightUpper.max(faceUppers[face])
                let e = leftUpper.sub(leftLower)
                cumLeft[i] = 2 * (e.x*e.y + e.x*e.z + e.y*e.z)
                e = rightUpper.sub(rightLower)
                cumRight[nfaces-i-1] = 2 * (e.x*e.y + e.x*e.z + e.y*e.z)
            }
            for (const i of range(nfaces-1)) {
                const below = cumLeft[i] / cumRight[0]
                const above = cumRight[i] / cumRight[0]
                const cost = .125 * (below * i + above * (nfaces - i))
                if (isNaN(cost)) throw new Error("NAN")
                if (cost <= best.cost) {
                    best.cost = cost
                    best.idx = i
                    best.axis = axis
                }
            }
        }
        if (best.axis != 2)
            faces.sort(cmpfn(best.axis))
        node.left = { faces: faces.subarray(0, best.idx) }
        node.right = { faces: faces.subarray(best.idx, nfaces) }
    }

    traceRay(start, dir) {
        const { verts, tris } = this
        let result = { t: Infinity }

        const recurse = (node) => {
            if (!node.leaf) {
                const children = [node.left, node.right]
                const dist = children.map(child => intersectRayAABB(start, dir, child.lower, child.upper))
                const [closest, furthest] = dist[1] < dist[0] ? [1,0] : [0,1]
                if (dist[closest] < result.t)
                    recurse(children[closest])
                if (dist[furthest] < result.t)
                    recurse(children[furthest])                
            } else {
                for (const face of node.faces) {
                    const tri = tris[face]
                    const a = verts[tri[0].vidx], b = verts[tri[1].vidx], c = verts[tri[2].vidx]
                    const t = intersectRayTri(start, dir, a, b, c)
                    if (t < result.t) {
                        result.t = t
                        result.tidx = face
                    }
                }
            }
        }
        recurse(this.top)
        return result
    }
        
}

function assert(cond) {
    if (!cond) throw new Error()
}


module.NN = class NN {


    constructor(points, r) {

        const d = 2*r
        const zero = Vec3.of(0), cmax = Vec3.of(NBINS-1)
        const NBINS = 255, NBINS2 = NBINS**2, NBINS3=NBINS**3
        
        const counts = new Uint32Array(NBINS3)
        const hashes = new Int32Array(points.length)
        for (let i = 0; i < points.length; i++) {
            let c = points[i]
            c = c.divc(d)
            c = c.addc(NBINS/2)
            c = c.floor()
            c = c.max(zero)
            c = c.min(cmax)
            const hash = c.x + c.y * NBINS + c.z * NBINS2
            hashes[i] = hash
            counts[hash]++


        }
        let hmax = -Infinity, hmin = Infinity
        for (const hash of hashes) {
            hmax = max(hash,hmax)
            hmin = min(hash,hmin)
        }
        
        for (let hash = 1; hash < NBINS3; hash++)
            counts[hash] += counts[hash-1]

        const sorted = new Uint32Array(points.length)
        
        for (let i = 0; i < points.length; i++)
            sorted[--counts[hashes[i]]] = i
        
        Object.assign(this, { points, hashes, counts, sorted, d })

    }

    collisions2() {
        const { points, sorted, counts, hashes, d } = this
        const NBINS = 255, NBINS2 = NBINS**2, NBINS3=NBINS**3
        const dsq = d**2
        const matches = []
        for (let hash = 0; hash < NBINS3 - 1; hash++) {
            
        }
    }
    
    collisions() {

        const { points, sorted, counts, hashes, d } = this
        const NBINS = 255, NBINS2 = NBINS**2, NBINS3=NBINS**3
        const dsq = d**2
        const matches = []
        for (let p1idx = 0; p1idx < points.length; p1idx++) {
            const p = points[p1idx]
            const hash = hashes[p1idx]
            const hz = floor(hash / NBINS2)
            const hy = floor((hash - hz*NBINS2) / NBINS)
            const hx = hash - hz*NBINS2 - hy*NBINS

            for (let x = max(0, hx - 1); x <= min(NBINS - 1, hx + 1); x++)
            for (let y = max(0, hy - 1); y <= min(NBINS - 1, hy + 1); y++)
            for (let z = max(0, hz - 1); z <= min(NBINS - 1, hz + 1); z++) {
                const ohash = x + y*NBINS + z*NBINS2
                if (ohash >= (NBINS3-1) || ohash < 0) continue       
                const stop = counts[ohash+1]
                for (let i = counts[ohash]; i < stop; i++) {
                    const p2idx = sorted[i];
                    if (p2idx == p1idx) continue
                    
                    if (p.distsq(points[p2idx]) < dsq)
                        matches.push([p1idx,p2idx])
                }
            }
        }
        return matches
        
    }

}






   
function test() {
    const N = 50000
    const R = 0.08
    const d = R*2
    const dsq = d*d    
    const hi = 10
    const lo = -10

    rand.seed = 462
    const points = []
    const fix = n => n //round(n*10000)/10000
    for(let i=0; i < N; i++) {
        const p = Vec3.of(fix(rand.f(lo,hi)), fix(rand.f(lo,hi)), fix(rand.f(lo,hi)))
        points.push(p)
    }

    function check(matches) {
        for (const m of matches)
            m.sort((a,b)=>a-b)

        matches.sort(([a1,a2],[b1,b2]) => a1 == b1 ? a2-b2 : a1-b1)
        const nonsame = matches.filter(([a,b]) => a != b)
        const same = matches.filter(([a,b]) => a == b)
        
        let strs = nonsame.map(([a,b]) => `${a}_${b}`)
        let set = new Set(strs)
        let missing = reference.filter(x => !set.has(x)).map(s=>s.split('_').map(n=>parseInt(n)))

        let uniq = [...set].map(s=>s.split('_').map(n=>parseInt(n)))

        const correct = [], incorrect = []
        for (const [i,j] of uniq) {
            if (points[i].distsq(points[j]) < dsq) 
                correct.push([i,j])
            else
                incorrect.push([i,j])
        }


        for (const [i,j] of incorrect)
            console.log('wrong',i,j, points[i].dist(points[j]))
        
        console.log(matches.length, 'matches returned,', missing.length, 'missing,', correct.length,'correct,', same.length, 'sames,', incorrect.length,'wrong,', matches.length-uniq.length-same.length, 'dupe')
        return missing
    }


    function perf(fn,label) {
        let t = performance.now()
        const matches = fn()
        t = (performance.now()-t)/1000
        console.log(`${label}: ${t.toFixed(6)}s`)
        const ret = check(matches)
        console.log()
        return ret
    }

    let reference = []
    for (let i = 0; i < points.length; i++) {
        const p = points[i]
        for (let j = i+1; j < points.length; j++)
            if (p.distsq(points[j]) < dsq)
                reference.push([i,j])
    }    

    reference = reference.map(([a,b]) => `${a}_${b}`)

    perf(() => {
        const nn = new zNN(points, R);
        return nn.collisions()
    }, 'znn')
    
        


}


        









