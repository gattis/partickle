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

    nearestOnTri(p,tri) {
        let [a,b,c] = tri
        let ab = b.sub(a), ac = c.sub(a), ap = p.sub(a), bp = p.sub(b), d1 = ab.dot(ap), d2 = ac.dot(ap)
        if (d1 <= 0 && d2 <= 0) return a    
        let d3 = ab.dot(bp), d4 = ac.dot(bp)
        if (d3 >= 0 && d4 <= d3) return b
        let vc = d1*d4 - d3*d2
        if (vc <= 0 && d1 >= 0 && d3 < 0)
            return a.add(ab.mulc(d1/(d1-d3)))
        let cp = p.sub(c), d5 = ab.dot(cp), d6 = ac.dot(cp)
        if (d6 >= 0 && d5 <= d6) return c    
        let vb = d5*d2 - d1*d6
        if (vb <= 0 && d2 >= 0 && d6 <= 0)
            return a.add(ac.mulc(d2/(d2-d6)))
        let bc = c.sub(b), va = d3*d6 - d5*d4
        if (va < 0 && d4-d3 > 0 && d5-d6 > 0)
            return b.add(bc.mulc((d4-d3)/(d4-d3+d5-d6)))    
        return a.add(ab.mulc(vb).add(ac.mulc(vc)).divc(va+vb+vc))
    }

    distToNode(p, node) {
        if (!node) return Infinity
        let dmin = node.min.sub(p), dmax = p.sub(node.max)
        return v3(max(dmin.x, 0, dmax.x), max(dmin.y, 0, dmax.y), max(dmin.z, 0, dmax.z)).mag()
    }

    signedDist(p) {
        const tris = this.tris, result = { d: Infinity }
        const recurse = node => {
            if (node.faces.length == 1) {
                const tri = tris[node.faces[0]]
                const nearest = this.nearestOnTri(p, tri)
                const d = p.dist(nearest)
                if (abs(d-result.d) < 0.00001) result.tris.push(tri)
                else if (d < result.d) result.tris = [tri]
                if (d < result.d) ([result.d, result.p] = [d, nearest])
            }
            let dists = [node.left, node.right].map(child => [child, this.distToNode(p, child)])
            dists.sort((a,b) => a[1] - b[1])
            for (const [child,dist] of dists)
                if (dist <= result.d)
                    recurse(child)  
        }
        recurse(this.root)
        if (result.d == Infinity) return Infinity
        let n = v3(0)
        for (let [a,b,c] of result.tris)
            n = n.add(b.sub(a).cross(c.sub(a)))
        let ap = p.sub(result.p)
        let d = result.d * sign(n.dot(ap))
        return d
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
        let cache = this.createObjectStore('cache', { autoIncrement:false, keyPath:'meshId' })
        this.createObjectStore('bitmaps', { autoIncrement: true })
        for (let store of [verts, faces, cache]) store.createIndex('meshId','meshId')
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
            for (const store of ['verts','faces','cache'])
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

