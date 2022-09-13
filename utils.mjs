const { cos, sin, acos, asin, cbrt, sqrt, PI, random, ceil, floor, tan, max, min, log2 } = Math

globalThis.I32MIN = -(2**31)
globalThis.I32MAX = 2**31-1
globalThis.MB = 2**20
globalThis.GB = 2**30

globalThis.roundUp = (n,k) => ceil(n/k)*k
globalThis.range = function* (a,b,step) {
    const [start,stop] = b == undefined ? [0,a] : [a,b]
    step ||= 1
    let val = start
    if (step >= 0)
        for (let val = start; val < stop; val += step) yield val
    else
        for (let val = start; val > stop; val += step) yield val
}

globalThis.enumerate = function* (arr) {
    for (const i of range(arr.length))
        yield [i,arr[i]]
}

globalThis.BitField = class BitField {
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

Array.prototype.sum = function(fn) {
    fn ||= x=>x
    return this.reduce((a,b) => a + fn(b), 0)
}


globalThis.coroutine = f => {
    var o = f();
    o.next();
    return function(x) {
        o.next(x);
    }
}

globalThis.clamp = (x,lo,hi) => x < lo ? lo : (x > hi ? hi : x)


globalThis.sleep = (s) => {
    return new Promise(done => setTimeout(done, s*1000))
}


globalThis.rand = {
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


globalThis.fetchtext = async (url) => {
    return await (await fetch(url)).text()
}


globalThis.Heap = class Heap {
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




























        
   
