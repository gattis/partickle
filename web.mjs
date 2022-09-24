//import { createRequire } from 'module';
//const require = createRequire(import.meta.url);
//const { GPU } = require('node-dawn')
//const fs = await import('fs')
//const sharp = (await import('sharp')).default

let gpu = navigator.gpu //new GPU(["dawn-backend=vulkan","disable-dawn-features=disallow_unsafe_apis"])

const blacklist = ['isFallbackAdapter','constructor','onuncapturederror','label']

const deepfake = (obj, remap, types) => {
    if (!(obj instanceof Object)) return obj
    if (obj instanceof Promise)
        return new Promise(resolve => {
            obj.then((...args) => {
                resolve(...args.map(arg => deepfake(arg, remap,types)))
            })
        })
    const prox = {}
    const name = obj.constructor.name
    if (!types.includes(name)) return obj
    for (const prop of Object.getOwnPropertyNames(obj.__proto__)) {
        if (blacklist.includes(prop)) continue
        const real = obj.__proto__[prop]
        if (!(real instanceof Function)) continue
        if (name in remap && real.name in remap[name]) {
            const fake = remap[name][real.name]
            prox[prop] = (...args) => {
                fake(obj,real,args)
            }                    
        } else {
            prox[prop] = (...args) => {
                args = args.map(arg => {
                    if (arg.real) return arg.real
                    for (var argprop in arg)
                        if (arg[argprop] && arg[argprop].real)
                            arg[argprop] = arg[argprop].real
                    return arg
                })
                let ret = real.call(obj, ...args)
                return deepfake(ret, remap, types)
            }
        }
    }
    for (const prop of Object.getOwnPropertyNames(obj)) {
        Object.defineProperty(prox, prop, {
            get() {
                return deepfake(obj[prop], remap, types)                
            },
            set(value) {
                obj[prop] = value
            }
        })                     
    }
    prox.real = obj
    return prox
}



const imgToTex = (obj,fn,args) => {
    const [source,dest,size] = args
    const image = source.source
    dest.texture = dest.texture
    const layout = { offset: 0, bytesPerRow: image.width * 4, rowsPerImage: image.height }
    obj.writeTexture(dest, image.data, layout, size)
}


//let fakegpu = deepfake(gpu, { GPUQueue: { copyExternalImageToTexture: imgToTex }}, ['GPU','GPUAdapter','GPUDevice','GPUQueue','WinCtx'])


let nframes = 0
let tstart = 0

Object.assign(navigator.gpu, {
    getPreferredCanvasFormat: ()=>'rgba8unorm'
})

Object.assign(globalThis, {
    //navigator: { gpu: fakegpu, userAgent: "nodejs" },
   // localStorage: {},   
    async fetch(path) {
        const text = fs.readFileSync('../particleboy/'+path, 'utf8')
        return new Promise(resolve => {
            resolve({ text() { return text } })
        })
    },
    requestAnimationFrame(callback) {
        queueMicrotask(() => {
            const t = performance.now()
            if (nframes++ == 0)
                tstart = t
            else if (nframes % 600 == 0)
                console.log('fps:', nframes*1000/(t - tstart))
            callback(t)
            navigator.ctx.refresh()
        })
    },
    Image: class {
        async decode() {}
    },
    async createImageBitmap(img) {
        const image = sharp(img.src)
        const { width, height } = await image.metadata()
        const data = await image.ensureAlpha().raw().toBuffer()
        return { width, height, data }
    }
})



/*
const a = await navigator.gpu.requestAdapter()
console.log(a)
const d = await a.requestDevice()
console.log(d)

const img = new Image()
img.src = './marble.png'
await img.decode()
const bitmap = await createImageBitmap(img)
const size = { width: bitmap.width, height: bitmap.height }
const fmt = navigator.gpu.getPreferredCanvasFormat()
const desc = { label: 'text', size, format: fmt, sampleCount:1,
           usage: GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.COPY_DST }

console.log(desc)
const texture = d.createTexture(desc)
console.log(navigator.gpu)
d.queue.copyExternalImageToTexture(
    { source: bitmap, origin: {x:0, y:0, z:0}, flipY: true },
    { texture, mipLevel: 0, origin: {x:0, y:0, z:0}, aspect: 'all', colorSpace:'srgb', premultipliedAlpha: false },
    [ bitmap.width, bitmap.height, 1]
)
console.log('done')*/


