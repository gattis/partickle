

const W = 1500, H = 1500

let adapter, nframes = 0, tstart = 0

await import('./web.mjs')

const { v3 } = await import ("./gpu.mjs")
const { Sim } = await import("./sim.mjs")

let move = false


navigator.ctx = navigator.gpu.createWindow(W, H, "particleboy", (type, ...args) => {
    const { compute, render, gpu, bufs, possessed, camera } = sim
    if (type == 'quit') {
        ctx.close()
        process.exit(0)
    }
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
})



const sim = new Sim()
await sim.init(W,H,navigator.ctx)
sim.run()






