import * as util from './utils.js'
import * as gpu from './gpu.js'
Object.assign(globalThis, gpu, util)

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

