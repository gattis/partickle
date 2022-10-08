import * as util from './utils.js'
import * as gpu from './gpu.js'
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
    nearHash(x,y,z) {
        return [[x+1,y,z],[x,y+1,z],[x,y,z+1],[x-1,y,z],[x,y-1,z],[x,y,z-1]]
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
            const ps = [0,1,2].map(i=>relverts[tri[i][0]])
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
                
        const R = D/2
        this.samples = []
        this.gradients = []
        this.edges = []
        for (const [v,i] of hashmap) {
            const [x,y,z] = v.split(',').map(istr => parseInt(istr))
            const edges = []
            for (const [xi,yi,zi] of this.nearHash(x,y,z)) {
                const id = hashmap.get(String([xi,yi,zi]))
                if (id != undefined) edges.push(id)
            }
            this.edges.push(edges)            
            this.samples.push(v3(c.x + x*D + R, c.y + y*D + R, c.z + z*D + R))
            this.gradients.push(normals[i].normalized())
        }
        
        console.log(`voxelize took ${performance.now() - tstart}ms`)

    }
}








