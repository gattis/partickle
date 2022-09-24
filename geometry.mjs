const { abs,cos,sin,acos,asin,cbrt,sqrt,pow,PI,random,round,ceil,floor,tan,max,min,log2 } = Math
import * as util from './utils.mjs'
import * as gpu from './gpu.mjs'
Object.assign(globalThis, gpu, util)

export const intersectRayAABB = (start, dir, lower, upper) => {
    const df = dir.recip()
    const tlo = lower.sub(start).mul(df)
    const thi = upper.sub(start).mul(df)
    let tmin = max(max(min(tlo.x,thi.x), min(tlo.y,thi.y)), min(tlo.z,thi.z))
    let tmax = min(min(max(tlo.x,thi.x), max(tlo.y,thi.y)), max(tlo.z,thi.z))
    if (tmax < 0 || tmin > tmax) return Infinity
    return tmin
}


export class VoxelGrid {
    constructor(verts, tris, D) {

        this.tris = tris
        this.c = v3(0);
        for (const vert of verts)
            this.c = this.c.add(vert)
        this.c = this.c.divc(verts.length)
        this.relverts = verts.map(vert=>vert.sub(this.c))
        this.hashmap = new Map()
        this.normals = []
        this.D = D
        this.lim = D*sqrt(2)/2
    }
    hash(p) {
        const v = p.divc(this.D*1.0001).floor()
        return String([v.x,v.y,v.z])
    }
    addPoint(p,n) {
        const { hashmap, normals } = this
        let hash = this.hash(p)
        let id = hashmap.get(hash)
        if (id == undefined) {
            id = normals.length
            normals.push(n)
            hashmap.set(hash, id)
        } else normals[id] = normals[id].add(n)
    }
    voxelize() {
        const { tris, relverts, D, lim, hashmap, c, normals } = this
        let tstart = performance.now()
        for (const [tidx,tri] of enumerate(tris)) {
            const ps = [0,1,2].map(i=>relverts[tri[i].vidx])
            const N = ps[1].sub(ps[0]).cross(ps[2].sub(ps[0])).normalized()
            const ls = [[1,2],[0,2],[0,1]].map(([a,b]) => ps[a].sub(ps[b]).mag())
            const order = [0,1,2]
            order.sort((a,b) =>  ls[a]-ls[b])
            const [A,B,C] = [1,2,0].map(k => ps[order[k]])
            const AB = B.sub(A), AC = C.sub(A)
            const b = AC.mag()
            const n = AC.cross(AB).normalized()
            const i = AC.normalized()
            const j = n.cross(i).normalized()
            const height = j.dot(AB)
            const split = i.dot(AB)
            const lslope = split/height
            const rslope = (b-split)/height
            const nj = ceil(height/lim)
            const dj = height / nj
            for (let row = 0; row <= nj; row++) {
                const jpos = row * dj
                const istart = jpos * lslope
                const li = b - jpos * rslope - istart
                const ni = ceil(li/lim)
                const di = ni == 0 ? 0 : li/ni
                for (let col = 0; col <= ni; col++) {
                    const ipos = istart + col*di
                    this.addPoint(A.add(i.mulc(ipos)).add(j.mulc(jpos)), N)
                }
            }
        }

        this.vertidxs = relverts.map((v,i) => hashmap.get(this.hash(v)))
        for (const idx of this.vertidxs)
            if (idx == undefined) {
                consle.error('all vertices should map to particles')
                debugger;
            }
            
        let iter = 0
        while (false) {
            const newsamples = []
            for (let [v,n] of grid) {
                const coord = v.split(',').map(i => parseInt(i))
                let axis = n.majorAxis()
            
                let dir = n[axis] < 0 ? 1 : -1
                coord[axis] += dir
                let hash = String(coord)
                let entry = grid.get(hash)
                if (entry) continue
                newsamples.push([hash,n])
            }
            if (newsamples.length == 0) break
            for (let [hash,n] of newsamples) {
                let entry = grid.get(hash) || v3(0)
                entry = entry.add(n)
                grid.set(hash,entry)
            }        
            if (++iter >= 20) break
        } 
        console.log('voxelize iters:', iter)
    
        const R = D/2
        this.samples = []
        this.gradients = []
        for (const [v,i] of hashmap) {
            const [x,y,z] = v.split(',').map(istr => parseInt(istr))
            this.samples.push(v3(c.x + x*D + R, c.y + y*D + R, c.z + z*D + R))
            this.gradients.push(normals[i].normalized())
        }
        
        //const sdf = SDF(voxels, dim)
        //const gradients = voxels.map(([x,y,z]) => sdfGrad(sdf, dim, x, y, z).normalized())
        console.log(`voxelize took ${performance.now() - tstart}ms`)

    }
}

export const sdfGrad = (sdf, dim, x, y, z) => {
    const dx = sampleGrid(sdf, dim, min(x + 1, dim.x - 1), y, z) - sampleGrid(sdf, dim, max(x - 1, 0), y, z)
    const dy = sampleGrid(sdf, dim, x, min(y + 1, dim.y - 1), z) - sampleGrid(sdf, dim, x, max(y - 1, 0), z)
    const dz = sampleGrid(sdf, dim, x, y, min(z + 1, dim.z - 1)) - sampleGrid(sdf, dim, x, y, max(z - 1, 0))
    const grad = v3(dx,dy,dz)
    return dim.divc(2).mul(grad)
}



export const sampleGrid = (voxgrid, dim, x, y, z) => {
    return voxgrid[clamp(x, 0, dim.x-1) + clamp(y, 0, dim.y-1)*dim.x + clamp(z, 0, dim.z-1)*dim.x*dim.y]
}

export const edgeDetect = (voxgrid, dim, x, y, z) => {
    const center = sampleGrid(voxgrid, dim, x, y, z)
    let dist = Infinity
    for (const k of [z - 1, z, z + 1])
        for (const j of [y - 1, y, y + 1])
	    for (const i of [x - 1, x, x + 1])
		if (sampleGrid(voxgrid, dim, i, j, k) != center)
		    dist = min(dist, sqrt((x-i)**2 + (y-j)**2 + (z-k)**2) / 2);
    return dist
}

export const SDF = (voxels, dim) => {

    for (const [x,y,z] of voxels)
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












