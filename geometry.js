import * as util from './utils.js'
import * as gpu from './gpu.js'
Object.assign(globalThis, gpu, util)


     
export const tetVolume = (a,b,c,d) => {
    const ab = b.sub(a), ac = c.sub(a), ad = d.sub(a)
    const vol = ab.cross(ac).dot(ad) / 6.0
    if (vol < 0) throw new Error("tet verts must be ordered a,b,c,d such that ((b-a)x(c-a)).(d-a) > 0")
    return vol
}
const det3 = (b,c,d) => b[0]*c[1]*d[2] + c[0]*d[1]*b[2] + d[0]*b[1]*c[2] - d[0]*c[1]*b[2] - c[0]*b[1]*d[2] - b[0]*d[1]*c[2]

export const pointInsideTet = (p,verts) => {
    const [a,b,c,d] = verts.map(v => v.sub(p))
    const [detA, detB, detC, detD] = [det3(b,c,d), det3(a,c,d), det3(a,b,d), det3(a,b,c)]
    return (detA > 0.0 && detB < 0.0 && detC > 0.0 && detD < 0.0) || (detA < 0.0 && detB > 0.0 && detC < 0.0 && detD > 0.0)
}

export class BVHTree {
    constructor(tris) {
        const bounds = Array(tris.length), centroids = Array.from(range(3)).map(k => new Float32Array(tris.length))
        for (const [tidx,[a,b,c]] of enumerate(tris)) {
            bounds[tidx] = [a.min(b).min(c), a.max(b).max(c)]
            const centroid = a.add(b).add(c).divc(3)
            for (const k of range(3)) 
                centroids[k][tidx] = centroid[k]
        }
        Object.assign(this, { root:{ faces:new Uint32Array(range(tris.length)) }, tris, bounds })
        const recurse = node => {
            let min = v3(Infinity)
            let max = v3(-Infinity)
            for (const face of node.faces) {
                const [bmin,bmax] = bounds[face]
                min = min.min(bmin)
                max = max.max(bmax)
            }
            Object.assign(node, { min, max })
            if (node.faces.length == 1) return
            this.partition(node, centroids[max.sub(min).majorAxis()])
            if (node.left) recurse(node.left)
            if (node.right) recurse(node.right)
        }
        recurse(this.root)
    }

    partition(node, vals) {
        const faces = node.faces, nfaces = faces.length, bounds = this.bounds
        faces.sort((a,b) => vals[a]-vals[b])
        const lCum = new Float32Array(nfaces), rCum = new Float32Array(nfaces)
        let lMin = v3(Infinity), lMax = v3(-Infinity)
        let rMin = v3(Infinity), rMax = v3(-Infinity)
        for (const i of range(nfaces)) {
            const lbounds = bounds[faces[i]]
            const rbounds = bounds[faces[nfaces-i-1]]
            lMin = lMin.min(lbounds[0])
            lMax = lMax.max(lbounds[1])
            rMin = rMin.min(rbounds[0])
            rMax = rMax.max(rbounds[1])
            let l = lMax.sub(lMin), r = rMax.sub(rMin)
            lCum[i] = 2 * (l.x*l.y + l.x*l.z + l.y*l.z)
            rCum[nfaces-i-1] = 2 * (r.x*r.y + r.x*r.z + r.y*r.z)
        }

        const invSA = 1 / rCum[0]
        let best = {idx: 0, cost: Infinity}
        for (const idx of range(nfaces-1)) {
            const cost = .125 * invSA * (lCum[idx] * idx + rCum[idx] * (nfaces - idx))
            if (cost <= best.cost)
                best = { idx, cost }
        }
               
        const left = faces.subarray(0, best.idx+1), right = faces.subarray(best.idx+1, nfaces)
        if (left.length > 0) node.left = { faces: left }
        if (right.length > 0) node.right = { faces: right }
    }

    rayToNode(start, dir, node) {
        if (!node) return Infinity
        const df = dir.recip()
        const tlo = node.min.sub(start).mul(df)
        const thi = node.max.sub(start).mul(df)
        let tmin = max(max(min(tlo.x,thi.x), min(tlo.y,thi.y)), min(tlo.z,thi.z))
        let tmax = min(min(max(tlo.x,thi.x), max(tlo.y,thi.y)), max(tlo.z,thi.z))
        if (tmax < 0 || tmin > tmax) return Infinity
        return tmin
    }

    rayToTri(start, dir, tri) {
        const [a,b,c] = tri
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

    traceRay(start, dir) {
        const tris = this.tris, result = { t: Infinity }
        const recurse = node => {
            if (node.faces.length == 1) {
                const tri = tris[node.faces[0]]
                const t = this.rayToTri(start, dir, tri)
                if (t < result.t) Object.assign(result, { t, tri })
                return
            }
            let dists = [node.left, node.right].map(child => [child, this.rayToNode(start, dir, child)])
            dists.sort((a,b) => a[1] - b[1])
            for (const [child,dist] of dists)
                if (dist < result.t)
                    recurse(child)         
        }
        recurse(this.root)
        return result
    }

    distToNode(p, node) {
        if (!node) return Infinity
        let dmin = node.min.sub(p), dmax = p.sub(node.max)
        return v3(max(dmin.x, 0, dmax.x), max(dmin.y, 0, dmax.y), max(dmin.z, 0, dmax.z)).mag()
    }

    nearestOnTri(p,tri) {
        let res, [a,b,c] = tri
        let ab = b.sub(a), ac = c.sub(a), bc = c.sub(b), ap = p.sub(a), bp = p.sub(b), cp = p.sub(c)
        let n = ab.cross(ac).normalized()
        let d1 = ab.dot(ap), d2 = ac.dot(ap)
        if (d1 <= 0 && d2 <= 0) return [a, n]
        let d3 = ab.dot(bp), d4 = ac.dot(bp)
        if (d3 >= 0 && d4 <= d3) return  [b, n]
        let vc = d1*d4 - d3*d2
        if (vc <= 0 && d1 >= 0 && d3 < 0)
            return  [a.add(ab.mulc(d1/(d1-d3))), n]
        let d5 = ab.dot(cp), d6 = ac.dot(cp)
        if (d6 >= 0 && d5 <= d6) return  [c, n]
        let vb = d5*d2 - d1*d6
        if (vb <= 0 && d2 >= 0 && d6 <= 0)
            return  [a.add(ac.mulc(d2/(d2-d6))), n]
        let va = d3*d6 - d5*d4
        if (va < 0 && d4-d3 > 0 && d5-d6 > 0)
            return  [b.add(bc.mulc((d4-d3)/(d4-d3+d5-d6))), n]
        return [a.add(ab.mulc(vb).add(ac.mulc(vc)).divc(va+vb+vc)), n]
    }



    signedDist(query) {
        query.d = Infinity
        const recurse = node => {
            if (node.faces.length == 1) {
                const tri = this.tris[node.faces[0]]
                let [ptri,n] = this.nearestOnTri(query.p, tri)
                let delta = query.p.sub(ptri)
                let d = delta.mag()
                if (abs(d - query.d) <= 0.00001) query.tris.push([ ptri, tri, n ])
                else if (d < query.d) query.tris = [[ ptri, tri, n ]]
                if (d < query.d) ([query.d, query.delta] = [d, delta])
            }
            let dists = [node.left, node.right].map(child => [child, this.distToNode(query.p, child)])
            dists.sort((a,b) => a[1] - b[1])
            for (const [child,dist] of dists)
                if (dist <= query.d + 0.00001)
                    recurse(child)
        }
        recurse(this.root)
        if (query.d == Infinity) return query
        let un = v3(0), wn = v3(0)
        for (let [ptri, tri, n] of query.tris) {
            let weight = 1
            if (query.tris.length > 2) {
                let [a,b,c] = [...tri].sort((a,b) => a.dist(ptri) - b.dist(ptri))
                weight = acos(b.sub(a).normalized().dot(c.sub(a).normalized()))
            }
            un = un.add(n)
            wn = wn.add(n.mulc(weight))
        }
        query.n = wn.normalized()
        query.sdf = query.d * sign(un.dot(query.delta))
        return query
    }

 
}

class GeoDB extends IDBDatabase {
    
    static async open() {
        const db = await new Promise(resolve => {
            let req = window.indexedDB.open('geo')
            req.on('success', e => resolve(req.result))
            req.on('error', e => { throw new Error(e) })
            req.on('upgradeneeded', e => {
                Object.setPrototypeOf(e.target.result, GeoDB.prototype)
                e.target.result.create()
            })
        })
        Object.setPrototypeOf(db, GeoDB.prototype)
        db.storeNames = [...db.objectStoreNames]
        globalThis.db = db
    }

    static async reset() {
        let olddb = globalThis.db
        globalThis.db = undefined
        olddb.close()     
        await new Promise(resolve => window.indexedDB.deleteDatabase('geo').on('success', resolve))
        await this.open()
    }

    create() {
        dbg('creating db')
        for (let store of this.objectStoreNames)
            this.deleteObjectStore(store)
        this.createObjectStore('meshes', { autoIncrement: true })
        let verts = this.createObjectStore('verts', { autoIncrement: true })
        let faces = this.createObjectStore('faces', { autoIncrement: true })
        this.createObjectStore('bitmaps', { autoIncrement: true })
        for (let store of [verts, faces]) store.createIndex('meshId','meshId')
        for (let i of range(3)) faces.createIndex(`vertId${i}`, `vertId${i}`)
    }
 
    async transact(stores, perm, cb) {
        let x = this.transaction(stores, perm, { durability:'relaxed' })
        Object.setPrototypeOf(x, GeoTransact.prototype)
        const result = await cb(x)
        x.commit()
        return result
    }



}
    


class GeoVert {
    constructor(id, pos) {
        Object.assign(this, {id, pos, emap:new Map(), edges:[]})
    }
    update_edges() {
        let { emap, edges } = this
        for (let edge = [...emap.values()][0]; edges.length < emap.size; edge = emap.get(edge.vert))
            edges.push(edge)
    }
}
    
class GeoFace {
    constructor(id, verts, uvs) {
        for (let i of range(verts.length))
            verts[i].emap.set(verts[mod(i+1, verts.length)], new GeoEdge(verts[mod(i-1, verts.length)], this))
        Object.assign(this, {id, verts, uvs, edges:[]})
    }
}

class GeoEdge {
    constructor(vert, face) {
        Object.assign(this, {vert, face})
    }
}

class GeoMesh {   
    constructor(verts, faces, uvs) {
        verts = verts.map((v,id) => new GeoVert(id, v3(...v)))
        faces = faces.map((vids,id) => new GeoFace(id, vids.map(vid => verts[vid]), uvs[id]))
        for (let vert of verts) vert.update_edges()
        Object.assign(this,{verts,faces})
    }

    dual() {
        let newVerts = [], newFaces = [], newUvs = []
        for (let face of this.faces)
            newVerts.push(face.verts[0].pos.add(face.verts[1].pos).add(face.verts[2].pos).divc(3))
        for (let vert of this.verts) {
            newFaces.push(vert.edges.map(edge => edge.face.id))
            newUvs.push(vert.edges.map(edge => [0,0]))
        }
        return new GeoMesh(newVerts, newFaces, newUvs)
    }


        /*fragment(n, expand) {
        let v0 = this.verts[0]
        let visited = new Set([v0])
        let q = [0]
        let frags = []
        let frag = []
        while q

        while len(queue) != 0:

        vidx1 = queue.pop(0)
        cluster.append(vidx1)
        if len(cluster) > nper:
        clusters.append(cluster)
        cluster = []
        v1 = bm.verts[vidx1]
        adj = [e.other_vert(v1).index for e in v1.link_edges]
        for vidx2 in adj:
        if vidx2 not in queued:
        queue.append(vidx2)
        queued.add(vidx2)
        clusters.append(cluster)

    }*/
    
    fragment(minVerts, expand) {
        if (this.verts.length < minVerts*2)
            return [{ prime: new Set(this.verts) }]
        let bmin = v3(Infinity), bmax = v3(-Infinity)
        for (let v of this.verts) {
            bmin = bmin.min(v.pos)
            bmax = bmax.max(v.pos)
        }
        let span = bmax.sub(bmin)        
        for (let s = span.minc() / 8; s <= span.maxc(); s *= 1.5) {
            let dim = span.divc(s).ceil()
            let smin = bmin.sub(dim.mulc(s).sub(span).divc(2))
            let frags = [...range(dim.x * dim.y * dim.z)].map(i => [])
            for (let v of this.verts) {
                let d = v.pos.sub(smin).divc(s*1.000001).floor()
                frags[d.x + d.y*dim.x + d.z*dim.x*dim.y].push(v)
            }
            frags = frags.filter(frag => frag.length > 0)
            if (!frags.some(frag => frag.length < minVerts)) {
                frags = frags.map(frag => ({ prime:new Set(frag), aux:new Set() }))
                if (expand > 0) {
                    for (let frag of frags) {
                        for (let v of frag.prime)
                            for (let e of v.edges)
                                if (!frag.prime.has(e.vert))
                                    frag.aux.add(e.vert)
                        for (let iter of range(expand-1))
                            for (let v of [...frag.aux])
                                for (let e of v.edges)
                                    if (!frag.prime.has(e.vert))
                                        frag.aux.add(e.vert)
                    }
                }
                return frags
            }
        }
        return [{ prime: new Set(this.verts) }]
    }
        
}



class GeoTransact extends IDBTransaction {


    async delete(storeName, args) {
        let store = this.objectStore(storeName)
        if (args.index == undefined) 
            return await store.delete(args.key)
        let index = store.index(args.index)
        let keys = await this.wait(index.getAllKeys(IDBKeyRange.bound(args.key,args.key)))
        for (let key of keys)
            await store.delete(key)
    }

    async deleteWithRelatives(store, key) {
        await this.delete(store, { key })
        if (store == 'meshes')
            for (const store of ['verts','faces'])
                await this.delete(store, { index:'meshId', key })
        else if (store == 'verts')
            for (const i of range(3))
                await this.delete('faces', { index:'vertId'+i, key })
    }

    async update(storeName, key, col, val) {
        let store = this.objectStore(storeName)
        let record = await this.wait(store.get(key))
        record[col] = val
        return await this.wait(store.put(record, key))
    }

    async query(store, args = {}) {
        let [keys,vals] = await Promise.all([
            this.op(store, { ...args, method:'getAllKeys' }),
            this.op(store, { ...args, method:'getAll' })
        ])
        const data = new Map()
        for (let i of range(keys.length))
            data.set(keys[i], vals[i])
        return data
    }

    async meshGeometry(meshId) {
        let [vertData,faceData] = await Promise.all([
            this.query('verts', { index:'meshId', key:meshId }),
            this.query('faces', { index:'meshId', key:meshId }),
        ])
        let vmap = {}
        let verts = []
        for (let [vidx,[vid,vert]] of enumerate(vertData)) {
            vmap[vid] = vidx
            verts.push(vert.pos)
        }
        let faces = [], uvs = []
        for (let [fid,face] of faceData) {
            let nverts = Object.keys(face).filter(key => key.startsWith('vertId')).length
            faces.push([...range(nverts)].map(i => vmap[face['vertId'+i]]))
            uvs.push(face.uv)
        }           
        return new GeoMesh(verts,faces,uvs)
    }
    
    op(store, args = {}) {
        let { index, key, method, startKey, count } = args
        let collection = this.objectStore(store)
        if (index) collection = collection.index(index)
        let keyRange = startKey ? IDBKeyRange.lowerBound(startKey, true) : IDBKeyRange.lowerBound(0)
        let methArgs = [key ? IDBKeyRange.bound(key,key) : keyRange]
        if (count != undefined) methArgs.push(count)
        return this.wait(collection[method].apply(collection, methArgs), args)
    }

    count(store, args = {}) {
        return this.op(store, { method:'count', ...args })
    }

    wait(req) {
        return new Promise(resolve => {
            req.on('success', e => resolve(req.result))
        })
    }

}




globalThis.GeoDB = GeoDB

