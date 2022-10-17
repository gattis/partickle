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

export const tetVolume = (a,b,c,d) => {
    const ab = b.sub(a), ac = c.sub(a), ad = d.sub(a)
    const vol = ab.cross(ac).dot(ad) / 6.0
    //if (vol < 0) throw new Error("tet verts must be ordered a,b,c,d such that ((b-a)x(c-a)).(d-a) > 0")
    return vol
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

const det3 = (b,c,d) => b[0]*c[1]*d[2] + c[0]*d[1]*b[2] + d[0]*b[1]*c[2] - d[0]*c[1]*b[2] - c[0]*b[1]*d[2] - b[0]*d[1]*c[2]

export const pointInsideTet = (p,verts) => {
    const [a,b,c,d] = verts.map(v => v.sub(p))
    const [detA, detB, detC, detD] = [det3(b,c,d), det3(a,c,d), det3(a,b,d), det3(a,b,c)]
    return (detA > 0.0 && detB < 0.0 && detC > 0.0 && detD < 0.0) || (detA < 0.0 && detB > 0.0 && detC < 0.0 && detD > 0.0)
}

export class GeoGrid {
    constructor(points, spacing) {
        let bounds = { hi: v3(-Infinity), lo: v3(Infinity) }
        for (const p of points) {
            bounds.hi = bounds.hi.max(p)
            bounds.lo = bounds.lo.min(p)
        }
        const ncells = bounds.hi.sub(bounds.lo).divc(spacing).ceil()
        const maxcell = ncells.subc(1).max(v3(1))
        const vhash = v3(1, ncells.x, ncells.x*ncells.y)
        const grid = new Map()
        for (const [pid,p] of enumerate(points)) {
            const hash = p.sub(bounds.lo).divc(spacing).floor().min(maxcell).mul(vhash).sum()
            const cell = grid.get(hash)
            if (cell == undefined) 
                grid.set(hash, [pid])
            else 
                cell.push(pid)
        }       
        Object.assign(this, { points, grid, bounds, maxcell, vhash, spacing })
     }

    within(p, dist) {
        const { points, grid, bounds, maxcell, vhash, spacing } = this
        const mincell = v3(0)
        const distsq = dist*dist
        const matches = []

        const start = p.subc(dist).sub(bounds.lo).divc(spacing).floor().min(maxcell).max(mincell)
        const stop = p.addc(dist).sub(bounds.lo).divc(spacing).floor().min(maxcell).max(mincell)
        for (let z = start.z; z <= stop.z; z++) 
            for (let y = start.y; y <= stop.y; y++) 
                for (let x = start.x; x <= stop.x; x++) {
                    const hash = vhash.mul([x,y,z]).sum()
                    const cell = grid.get(hash) || []
                    for (const pid of cell)
                        if (points[pid].distsq(p) < distsq) {
                            matches.push(pid) 
                        }
                }
        return matches
    }
}

