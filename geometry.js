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
        let show = p.dist([0.21213172376155853,0.21213172376155853,0.9292889833450317]) < 0.00001

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
            let req = window.indexedDB.open('geo', 1)
            req.on('success', e => resolve(req.result))
            req.on('error', e => { throw new Error(e) })
            req.on('upgradeneeded', e => {
                Object.setPrototypeOf(e.target.result, GeoDB.prototype)
                e.target.result.upgrade()
            })
        })
        Object.setPrototypeOf(db, GeoDB.prototype)
        return db
    }

    upgrade() {
        dbg('creating stores')
        this.waitCreate = new Promise(resolve => { this.doneCreate = resolve })
        this.createObjectStore('meshes', { autoIncrement: true })
        let verts = this.createObjectStore('verts', { autoIncrement: true })
        let faces = this.createObjectStore('faces', { autoIncrement: true })
        let particles = this.createObjectStore('particles', { autoIncrement: true })
        let tets = this.createObjectStore('tets', { autoIncrement: true })
        this.createObjectStore('bitmaps', { autoIncrement: true })
        for (let store of [verts, faces, particles, tets]) store.createIndex('meshId','meshId')
        for (let i of range(3)) faces.createIndex(`vertId${i}`, `vertId${i}`)
        for (let i of range(4)) tets.createIndex(`partId${i}`, `partId${i}`)
        dbg('stores created')
        this.doneCreate()
    }

    async reset() {
        const stores = [...this.objectStoreNames]
        this.transact(stores,'readwrite')
        await Promise.all(stores.map(store => this.wait(this.x.objectStore(store).clear())))
        this.commit()
    }

    async loadBitmap(name, data) {
        const img = new Image()
        img.src = data
        await img.decode()
        const bitmap = await createImageBitmap(img)
        let x = this.transaction(['bitmaps'], 'readwrite')
        let bitmaps = x.objectStore('bitmaps')
        await this.wait(bitmaps.add({ name, data: bitmap }))
        x.commit()
    }

    async loadWavefront(name, data) {
        this.transact([...this.objectStoreNames], 'readwrite')
        let meshes = this.x.objectStore('meshes')
        let meshId = await this.wait(meshes.add({
            name, bitmapId:-1, color:[1,1,1,1], offset:[0,0,0], rotation:[0,0,0], gravity:1, density:1,
            scale:[1,1,1], 'shape stiff':1, 'vol stiff':1, friction:1, 'collision damp':1, 'self collide':1
        }))
        let verts = this.x.objectStore('verts')
        let faces = this.x.objectStore('faces')
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
                vertIds[localVerts++] = await this.wait(verts.add({ pos, mass, meshId }))
            } else if (key == 'vt') {
                let vtdata = toks.map(parseFloat)
                uvIds[localUVs++] = vtdata.slice(0, 2)
            } else if (key == 'f') {
                if (toks.length == 3) {
                    let face = toks.map((tok,i) => [`vertId${i}`, vertIds[parseInt(tok.split('/')[0])]])
                    let uv = toks.map(tok => uvIds[parseInt(tok.split('/')[1])] || [0,0])
                    await this.wait(faces.add({ ...Object.fromEntries(face), uv, meshId }))
                } 
            }
        }
        this.commit()
    }




    async sampleMesh(meshId, D) {
        this.transact(['verts','faces'])
        const verts = await this.query('verts', { index:'meshId', key:meshId })
        const faces = await this.query('faces', { index:'meshId', key:meshId })
        const tris = Array.from(faces).map(([id,face]) => ['vertId0','vertId1','vertId2'].map(col=>v3(...verts.get(face[col]).pos)))
        const tree = new BVHTree(tris)
        let bmin = [Infinity,Infinity,Infinity], bmax = [-Infinity,-Infinity,-Infinity]
        for (const [vertId, vert] of verts)
            for (let k of [0,1,2]) {
                bmin[k] = min(vert.pos[k], bmin[k])
                bmax[k] = max(vert.pos[k], bmax[k])
            }
        let bounds = [0,1,2].map(k => (bmax[k]*10 - bmin[k]*10)/10)        
        dbg({bounds})
        let dim = [0,1,2].map(k => ceil(bounds[k]/D)).map(d => d + (d % 2 == 0 ? 2 : 1))
        dbg({dim})
        let space = [0,1,2].map(k => ((dim[k]-1)*D - bounds[k]) / 2)
        dbg({space})
        let offset = [0,1,2].map(k => bmin[k] - space[k])
        dbg({offset})
        let [dimx,dimy,dimz] = dim
        let dimxy = dimx*dimy

        let tetsA = [[6,3,5,0], [4,6,5,0], [2,3,6,0], [1,5,3,0], [6,5,3,7]]
        let tetsB = [[1,4,2,0], [1,2,4,7], [7,2,4,6], [4,1,7,5], [2,7,1,3]]

        let hpmap = new Map(), hvmap = new Map(), particles = [], tets = [], h = 0
        for (let [h,[x,y,z]] of enumerate(range3d(...dim))) {
            let p = v3(D*x,D*y,D*z).add(offset)
            if (tree.signedDist(p) <= D/10)
                hpmap.set(h, p)
        }
        for (let [xi,yi,zi] of range3d(...dim))
            for (let reltet of (xi+yi+zi) % 2 == 1 ? tetsB : tetsA) {
                let xyzs = reltet.map(vid => [xi + (vid&1), yi + Number(Boolean(vid&2)), zi + Number(Boolean(vid&4))])
                if (xyzs.some(([x,y,z]) => x >= dimx || y >= dimy || z >= dimz)) continue
                let hs = xyzs.map(([x,y,z]) => x + y*dimx + z*dimxy)
                let hps = hs.map(h => [h, hpmap.get(h)])
                if (hps.some(([h,p]) => p == undefined)) continue
                tets.push(hps.map(([h,p]) => {
                    let v = hvmap.get(h)
                    if (v == undefined) {
                        v = particles.length
                        particles.push(p)
                        hvmap.set(h, v)
                    }
                    return v
                }))
            }

        dbg({particles:particles.length})
        dbg({tets:tets.length})
        this.commit()

        await this.delete('particles', { index:'meshId', key:meshId })
        await this.delete('tets', { index:'meshId', key:meshId })

        this.transact(['particles','tets'],'readwrite')
        let partStore = this.x.objectStore('particles'), tetStore = this.x.objectStore('tets')
        const partIds = []
        for (let [i,p] of enumerate(particles))
            partIds[i] = await this.wait(partStore.add({ pos: [p.x,p.y,p.z], meshId }))
        for (let [a,b,c,d] of tets)
            await this.wait(tetStore.add({ partId0: partIds[a], partId1: partIds[b], partId2: partIds[c], partId3: partIds[d], meshId }))
        this.commit()
    }

    transact(stores, perm = 'readonly', options = {}) {
        if (this.x != undefined) throw Error('tranaction already open')
        options.durability ||= 'relaxed'
        this.x = this.transaction(stores, perm, options)
    }

    commit(stores, perm) {
        if (this.x == undefined) throw Error('no transaction was open')
        this.x.commit()
        this.x = undefined
    }

    wait(req) {
        return new Promise(resolve => {
            req.on('success', e => resolve(req.result))
        })
    }

    op(store, args = {}) {
        let { index, key, method, startKey, count } = args
        let collection = this.x.objectStore(store)
        if (index) collection = collection.index(index)
        let keyRange = startKey ? IDBKeyRange.lowerBound(startKey, true) : IDBKeyRange.lowerBound(0)
        let methArgs = [key ? IDBKeyRange.bound(key,key) : keyRange]
        if (count != undefined) methArgs.push(count)
        return this.wait(collection[method].apply(collection, methArgs), args)
    }

    count(store, args = {}) {
        return this.op(store, { method:'count', ...args })
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

    curse(store, args) {
        let { index, key, startKey, cb } = args
        let collection = this.x.objectStore(store)
        if (index) collection = collection.index(index)
        let keyRange = startKey ? IDBKeyRange.lowerBound(startKey, true) : IDBKeyRange.lowerBound(0)
        key = key == undefined ? keyRange : IDBKeyRange.bound(key,key)
        const req = collection.openCursor(key)
        return new Promise(resolve => {
            req.on('success', e => { 
                if (!e.target.result) resolve()
                else {
                    cb(e.target.result)
                    e.target.result.continue()
                }
            })
        })
    }

    async update(store, args) {
        let xnew = this.x == undefined
        if (xnew) this.transact([store], 'readwrite')
        await this.curse(store, { ...args, cb: cur => { 
            if (!cur.value) return
            cur.value[args.col] = args.val
            cur.update(cur.value)
        }})
        if (xnew) this.commit()
    }

    async delete(store, args) {
        let xnew = this.x == undefined
        if (xnew) this.transact([store], 'readwrite')
        await this.curse(store, { ...args, cb: cur => cur.delete() })
        if (xnew) this.commit()
    }

    async deleteWithRelatives(store, key) {
        await this.delete(store, { key })
        if (store == 'meshes')
            for (const store of ['verts','faces','particles','tets'])
                await this.delete(store, { index:'meshId', key })
        else if (store == 'verts')
            for (const i of range(3))
                await this.delete('faces', { index:'vertId'+i, key })
        else if (store == 'particles')
            for (const i of range(4))
                await this.delete('tets', { index:'partId'+i, key })    
    }
  
}




globalThis.GeoDB = GeoDB

