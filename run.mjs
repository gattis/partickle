import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const dawn = require('./dawn.node')
const fs = await import('fs')


const W = 1500, H = 1500
const gpu = dawn.create(["dawn-backend=vulkan","disable-dawn-features=disallow_unsafe_apis"])
let adapter, nframes = 0, tstart = 0

Object.assign(globalThis, {
    navigator: {
        userAgent: "nodejs",
        gpu: {
            async requestAdapter(...args) {
                adapter = await gpu.requestAdapter(...args)            
                return adapter
            },
            getPreferredCanvasFormat() {
                return 'bgra8unorm'
            }
        }
    },
    localStorage: {},   
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
            adapter.swapChainPresent()
            updateUI(adapter.pollUI())
        })
    }
})

const { v3 } = await import ("./gpu.mjs")
const { Sim } = await import("./sim.mjs")

const ctx = {
    configure() {},
    getCurrentTexture() {
        return {
            createView() {
                return adapter.swapChainTextureView()
            }
        }
    }
}

const sim = new Sim()
sim.init(W,H,ctx).then(() => {
    sim.run()
})

let move = false

function updateUI(eventstr) {
    const { compute, render, gpu, bufs, possessed, camera } = sim
    
    let events = eventstr.split('|')
    for (const event of events) {
        if (!event) continue
        let [type,...args] = event.split(":")
        args = args.map(arg=>parseFloat(arg))
        //console.log(type,...args)

        if (type == 'quit')        
            process.exit(0)
        
        if (type == 'key' && args[0] == 83) 
            compute.paused = !compute.paused

        if (type == 'key' && args[0] == 80) 
            compute.fwdstep = true

        if (type == 'mouseButton' && args[1] == 1) 
            move = { btn: args[0] }

        if ((type == 'cursorEnter' && args[0] == 0) || (type == 'mouseButton' && args[1] == 0))
            move = false

        if (type == 'cursorPos' && move) {
            const posx = args[0], posy = args[1]
            if (move.x == undefined) {
                move.x = posx
                move.y = posy
            }
            const dx = .01*(posx - move.x), dy = .01*(move.y - posy)
            move.x = posx
            move.y = posy
            if (move.btn == 0) {
                sim.camLR += dx
                sim.camUD = clamp(sim.camUD + dy,-PI/2, PI/2)
            } else if (move.btn == 2) {
                const delta = v3( -dx * cos(sim.camLR), dx * sin(sim.camLR), -dy);
                sim.camPos = sim.camPos.add(delta)
            }
        }
        
        if (type == 'scroll') {
            const dy = .1 * args[1]
            const camDir = sim.camFwd()
            sim.camPos.x += dy * camDir.x
            sim.camPos.y += dy * camDir.y
            sim.camPos.z += dy * camDir.z
        }
    }
}
