const module = globalThis
const { abs,cos,sin,acos,asin,cbrt,sqrt,pow,PI,random,round,ceil,floor,tan,max,min,log2 } = Math
import './gpu.js'





module.intersectRayAABB = (start, dir, lower, upper) => {
    const df = dir.recip()
    const tlo = lower.sub(start).mul(df)
    const thi = upper.sub(start).mul(df)
    let tmin = max(max(min(tlo.x,thi.x), min(tlo.y,thi.y)), min(tlo.z,thi.z))
    let tmax = min(min(max(tlo.x,thi.x), max(tlo.y,thi.y)), max(tlo.z,thi.z))
    if (tmax < 0 || tmin > tmax) return Infinity
    return tmin
}

const tolerance = 0.01

module.voxelize = (verts, tris, D) => {
    let tstart = performance.now()
    let R = D/2
    let lo = Vec3.of(Infinity), hi = Vec3.of(-Infinity)
    for (const v of verts) {
        lo = Vec3.min(lo, v)
        hi = Vec3.max(hi, v)
    }
    let axes = [0,1,2]
    let dim = axes.map(k => max(1,floor((hi[k]-lo[k])/D)))
    let [xdim, ydim, zdim] = dim
    let [xmax, ymax, zmax] = axes.map(k => dim[k] - 1)
    let [xlo, ylo, zlo] = axes.map(k => (lo[k] + hi[k] - dim[k]*D)/2)
    const relverts = verts.map(v => v.sub(Vec3.of(xlo,ylo,zlo)))
    const voxgrid = new Uint8Array(xdim * ydim * zdim)
    const triMap = new Map()
    const addVox = (x,y,z,tidx) => {
        const hash = clamp(x,0,xmax) + clamp(y,0,ymax)*xdim + clamp(z,0,zmax)*xdim*ydim
        voxgrid[hash] = 255
        let tidxs = triMap.get(hash) || []
        tidxs.push(tidx)
        triMap.set(hash, tidxs)
    }
    const cross = (a,b) => a[0]*b[1]-a[1]*b[0]
    const normals = []
    for (const [tidx,tri] of enumerate(tris)) {            
        const [A,B,C] = [0,1,2].map(i=>relverts[tri[i].vidx])
        const N = B.sub(A).cross(C.sub(A)).normalized()
        normals.push(N)
        const axis = N.abs().maxaxis()
        for (const P of [A,B,C])
            addVox(floor(P.x/D), floor(P.y/D), floor(P.z/D), tidx)
        const [a,b,c] = [A.toarray(), B.toarray(), C.toarray()]
        const [az] = a.splice(axis,1), [bz] = b.splice(axis,1), [cz] = c.splice(axis,1)
        const axb = cross(a,b), cxa = cross(c,a)
        const area = cross(b,c) + axb + cxa
        let u,v,w
        const [xi, yi] = [floor(min(a[0],b[0],c[0])/D), floor(min(a[1],b[1],c[1])/D)]
        const [xf, yf] = [floor(max(a[0],b[0],c[0])/D), floor(max(a[1],b[1],c[1])/D)]
        for (let x = xi; x <= xf; x++)
            for (let y = yi; y <= yf; y++) {
                const p = [x*D, y*D]
                if ((v = (cxa + cross(p,c) + cross(a,p))/area) < -tolerance) continue
                if ((w = (axb + cross(p,a) + cross(b,p))/area) < -tolerance) continue
                if ((u = 1 - v - w) < -tolerance) continue
                const s = [x,y]
                s.splice(axis,0,floor((u*az + v*bz + w*cz)/D))
                addVox(...s, tidx)
            }
    }
    
    



    const samples = [], gradients = []
    for (const x of range(xdim))
         for (const y of range(ydim))
             for (const z of range(zdim)) {
                 const hash = x + y*xdim + z*xdim*ydim
                 if (voxgrid[hash]) {
                     samples.push(Vec3.of(xlo + x*D+R, ylo + y*D+R, zlo + z*D+R))
                     const tidxs = triMap.get(hash)
                     let N = Vec3.of(0)
                     for (const tidx of tidxs)
                         N = N.add(normals[tidx])
                     gradients.push(N.divc(tidxs.length))
                     
                 }
             }

                
    //const sdf = SDF(voxels, dim)
    //const gradients = voxels.map(([x,y,z]) => sdfGrad(sdf, dim, x, y, z).normalized())
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

module.SDF = (voxels, dim) => {

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


function assert(cond) {
    if (!cond) throw new Error()
}








