
for (const prop of Object.getOwnPropertyNames(Math))
    globalThis[prop] = Math[prop]

globalThis.round = (x,n = 0) => Math.round(x * 10**n) / 10**n
export const roundUp = (n,k) => ceil(n/k)*k
export const roundUpPow = (n,e) => e ** ceil(log(n)/log(e))
export const I32MIN = -(2**31)
export const I32MAX = 2**31-1

export const range = function* (a,b,step) {
    const [start,stop] = b == undefined ? [0,a] : [a,b]
    step ||= 1
    let val = start
    if (step >= 0)
        for (let val = start; val < stop; val += step) yield val
    else
        for (let val = start; val > stop; val += step) yield val
}

export const enumerate = function* (arr) {
    for (const i of range(arr.length))
        yield [i,arr[i]]
}

export const BitField = class BitField {
    constructor(nbits) {
        this.nbits = nbits
        this.nbytes = ceil(nbits/8)
        this.data = new Uint8Array(this.nbytes)
        return new Proxy(this,this)
    }
    get(o,k) {
        if (isNaN(k)) return Reflect.get(o,k)
        const byte = floor(k/8), bit = k % 8
        return (o.data[byte] >> bit) & 1
    }
    
    set(o,k,v) {
        if (isNaN(k)) return Reflect.set(o,k,v)
        const byte = floor(k/8), bit = k % 8
        o.data[byte] = (o.data[byte] & ~(1 << bit)) | (Number(v) << bit)
        return true
    }
    
}



String.prototype.interp = function(args) {
    const keys = Object.keys(args), vals = Object.values(args).map(val => JSON.stringify(val));
    return eval(`((${keys.join(',')}) => \`${this}\`)(${vals.join(',')})`)
}

Array.prototype.sum = function(fn, init) {
    fn ||= x=>x
    if (init == undefined) init = 0
    return this.reduce((a,b) => a + fn(b), init)
}

Array.prototype.max = function() {
    return this.reduce((a,b) => max(a,b), -Infinity)
}

Array.prototype.min = function() {
    return this.reduce((a,b) => min(a,b), Infinity)
}


export const coroutine = f => {
    var o = f();
    o.next();
    return function(x) {
        o.next(x);
    }
}

export const clamp = (x,lo,hi) => x < lo ? lo : (x > hi ? hi : x)


export const sleep = (s) => {
    return new Promise(done => setTimeout(done, s*1000))
}


export const rand = {
    seed: Math.floor(Math.random()*(2**32-1)),
    next:  () => {
        let t = rand.seed += 0x6D2B79F5;
        t = Math.imul(t ^ t >>> 15, t | 1);
        t ^= t + Math.imul(t ^ t >>> 7, t | 61);
        return (t ^ t >>> 14) >>> 0
    },
    f: (lo,hi) => lo + (rand.next() / 4294967295) * (hi-lo),
    i: (lo,hi) => lo + (rand.next() % (hi-lo))
}


export const fetchtext = async (url) => {
    return await (await fetch(url)).text()
}


export const Heap = class Heap {
    constructor(cmp) {
        const items = this.items = []
        this.cmp = (i, j) => cmp(items[i], items[j])
        this.swap = (i, j) => [items[i], items[j]] = [items[j], items[i]]
    }

    push(value) {
        const { items, cmp, swap } = this
        let index = items.push(value) - 1
        while (true) {
            const parent = ceil(index/2 - 1)
            if (parent < 0 || cmp(index, parent) >= 0) break
            swap(parent, index)
            index = parent
        }

    }

    pop() {
        const { items, cmp, swap } = this
        if (items.length == 1) return items.pop()
        const ret = items[0]
        items[0] = items.pop()
        let index = 0
        while (true) {
            const left = 2*index + 1, right = left + 1
            const child = right < items.length && cmp(left, right) > 0 ? right: left
            if (left >= items.length || cmp(child, index) >= 0) break
            swap(index, child)
            index = child
        }
        return ret
    }
}


export const hijack = (cls, meth, replacement) => {
    cls.prototype[meth] = new Proxy(cls.prototype[meth], { apply: replacement })
}




export class Preferences {
    constructor(name) {
        this.name = name
        this.data = JSON.parse(localStorage[name] || '{}')
        this.keys = []
        this.keyWatch = {}
        this.ival = {}
        this.lo = {}
        this.hi = {}
        this.step = {}
        this.opts = {}
        this.type = {}
    }
    addNum(key, ival, lo, hi, step) {
        this.lo[key] = lo
        this.hi[key] = hi
        this.step[key] = step
        this.type[key] = 'num'
        this.addField(key,ival)
    }
    addChoice(key, ival, opts) {
        this.opts[key] = opts
        this.type[key] = 'choice'
        this.addField(key,ival)
    }
    addBool(key, ival) {
        this.type[key] = 'bool'
        this.addField(key,ival)
    }
    addField(key,ival) {
        this.keys.push(key)
        this.ival[key] = ival
        this.keyWatch[key] = []
        Object.defineProperty(this, key, { 
            get: () => this.getval(key),
            set: (val) => this.setval(key,val)
        })
        if (!(key in this.data)) this[key] = ival        
    }
    getval(key) {
        return this.data[key]
    }
    setval(key,val) {
        this.data[key] = val
        localStorage[this.name] = JSON.stringify(this.data)
        for (const cb of this.keyWatch[key])
            cb(val)
    }
    watch(keys, callback) {
        for (const key of keys)
            this.keyWatch[key].push(callback)
    }
}






















        
   
