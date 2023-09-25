
for (const prop of Object.getOwnPropertyNames(Math))
    globalThis[prop] = Math[prop]

globalThis.round = (x,n = 0) => Math.round(x * 10**n) / 10**n
export const roundUp = (n,k) => ceil(n/k)*k
export const roundUpPow = (n,e) => e ** ceil(log(n)/log(e))
export const I32MIN = -(2**31)
export const I32MAX = 2**31-1
export const EPS = 1e-6
export const roundEps = x => round(1e6 * x) / 1e6
export const range = function* (a,b,step) {
    const [start,stop] = b == undefined ? [0,a] : [a,b]
    step ||= 1
    let val = start
    if (step >= 0)
        for (let val = start; val < stop; val += step) yield val
    else
        for (let val = start; val > stop; val += step) yield val
}

export const range3d = function* (x,y,z) {
    for (const zi of range(z))
        for (const yi of range(y))
            for (const xi of range(x))
                yield [xi,yi,zi]
}

export const scope = (cb) => {
    return new Proxy({}, {
        get(target,prop) {
            if (prop != 'then')
                try { return cb(prop) }
                catch(err) {
                    if (!(err instanceof ReferenceError))
                        throw err
                }
            return Reflect.get(...arguments)
        }
    })
}


export const enumerate = function* (iterable) {
    if (iterable[Symbol.iterator]) {
        let i = 0
        for (const val of iterable)
            yield [i++,val]
    } else {
        for (let i of range(iterable.length))
            yield [i, iterable[i]]
    }
}

export const reversed = function* (arr) {
    if (!(arr instanceof Array)) arr = [...arr]
    for (let i = arr.length - 1; i >= 0; i--) yield arr[i];
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

EventTarget.prototype.on = function(types, fn, options = {}) {
    if (!(types instanceof Array)) types = [types]
    for (let type of types) {
        if (this.cbs?.[type]) this.removeEventListener(type, this.cbs[type])
        this.addEventListener(type, (this.cbs||={})[type] = fn, options)
    }
    return this
}

EventTarget.prototype.off = function(types, fn, options = {}) {
    if (!(types instanceof Array)) types = [types]
    for (let type of types) {
        if (this.cbs?.[type]) this.removeEventListener(type, this.cbs[type])
    }
    return this
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

Array.prototype.max = function(key = x => x) {
    return this.reduce((a,b) => key(a) >= key(b) ? a : b, this[0])
}

Array.prototype.min = function(key = x => x) {
    return this.reduce((a,b) => key(a) <= key(b) ? a : b, this[0])
}

Array.prototype.uniq = function () {
    return [...new Set(this)];
}

range.prototype.map = function (fn) { return [...this].map(fn) }
range.prototype.min = function (fn) { return [...this].min(fn) }
range.prototype.max = function (fn) { return [...this].max(fn) }



export const mod = (n, m) => ((n % m) + m) % m


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
    seed: 666,
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

export const hijack = (cls, meth, replacement) => {
    cls.prototype[meth] = new Proxy(cls.prototype[meth], { apply: replacement })
}

export class Preferences {
    constructor(name) {
        Object.assign(this, {
            name, keys:[], keyWatch:{}, ival:{}, lo:{}, hi:{}, step:{}, opts:{}, type:{}, hidden:{},
            data: JSON.parse(localStorage[name] || '{}')
        })
    }
    addNum(key, ival, lo, hi, step) {
        this.lo[key] = lo
        this.hi[key] = hi
        this.step[key] = step
        this.addField(key, ival, 'num')
    }
    addChoice(key, ival, opts) {
        this.opts[key] = opts
        this.addField(key, ival, 'choice')
    }
    addBool(key, ival) {
        this.addField(key, ival, 'bool')
    }
    addVector(key, ivec) {
        this.addField(key, ivec, 'vec')
    }
    addField(key,ival,type) {
        this.type[key] = type
        this.keys.push(key)
        this.ival[key] = ival
        this.keyWatch[key] = []
        Object.defineProperty(this, key, {
            get: () => this.getval(key),
            set: (val) => this.setval(key,val)
        })
        if (!(key in this.data)) this[key] = ival
        else if (type == 'vec')
            this.data[key] = v3(this.data[key][0], this.data[key][1], this.data[key][2])
    }
    getval(key) {
        return this.data[key]
    }
    setval(key,val) {
        if (this.data[key] == val) return
        this.data[key] = val
        localStorage[this.name] = JSON.stringify(this.data)
        for (const cb of this.keyWatch[key])
            cb(key,val)
    }
    watch(keys, callback) {
        for (const key of keys)
            this.keyWatch[key].push(callback)
    }
}

Map.prototype.getDefault = function(key, def) {
    const val = this.get(key)
    return val == undefined ? def : val
}

Map.prototype.setDefault = function(key, def) {
    const val = this.getDefault(key, def)
    this.set(key, val)
    return val
}

export const hashPair = (a,b) => (new Float64Array((new Uint32Array(a <= b ? [a,b] : [b,a])).buffer))[0]
export const unhashPair = hash => [...new Uint32Array((new Float64Array([hash])).buffer)]

export const randomShuffle = (arr) => {
    if (!(arr instanceof Array)) arr = [...arr]
    let g = 279470273, m = 2**31 - 1
    return arr.sort((a,b) => ((a+1)*g%m - (b+1)*g%m))
}

export const repr = (v) => {
    if (v == undefined) return 'undef'
    if (v == null) return 'null'
    if (typeof v == 'string') return v
    if (typeof v == 'number') {
        if (v%1 == 0) return v.toString()
        let [f,p] = v.toPrecision(6).split('e')
        f = f.includes('.') ? f.replace(/0+$/,'').replace(/\.$/,'') : f
        p = p ? 'e'+p : ''
        return f + p
    }
    if (v instanceof Array | v instanceof Set) {
        if ((v.length || v.size || Infinity) <= 8)
            return `[${[...v].map(x => repr(x)).join(',')}]`
        else return `[sz=${v.length || v.size || 0}]`
    }
    if (v.message) { return v.message }
    return v.toString()
}



export const dbg = (vars, indent = 0) => {
    let [fname,lineno] = new Error().stack.split('\n')[2].split('/').at(-1).split(':').slice(0,2)
    let prefix = [
        ' '.repeat(indent*4) + `%c${fname}|${lineno}:%c`,
        'color: yellow; font-weight:bold;', 'color: unset; font-weight:unset;'
    ]
    if (typeof vars == 'string') {
        console.out(...prefix,vars)
    } else {
        let reprs = Object.entries(vars).map(([k,v]) => `${k}=${repr(v)}`)
        console.groupCollapsed(...prefix, reprs.join(' '))
        for (let [k,v] of Object.entries(vars)) {
            console.group(k)
            console.dir(v)
            console.groupEnd()
        }
        console.groupEnd()
    }
}

console.out = console.log


















