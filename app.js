const { Sim, Params, render, phys } = await import('./sim.js')
window.render = render
window.phys = phys
const doc = document
window._ = sel => doc.querySelector(sel)
window.__ = sel => doc.querySelectorAll(sel)

window.cv = doc.querySelector('canvas')
cv.width = cv.style.width = window.innerWidth
cv.height = cv.style.height = window.innerHeight
cv.style.cursor = 'grab'

window.ctx = cv.getContext('webgpu')
window.sim = new Sim()
await sim.init(cv.width, cv.height, ctx)
const { possessed, gpu, bufs, camera, particles } = sim

const createCtrl = {
    'common': (prefs,key,elem,ctrl,out) => {
        if (out) _(elem).append(out)
        _(elem).append(ctrl)
        ctrl.setAttribute('id', key)
        ctrl.setAttribute('name', key)
        ctrl.setAttribute('value', prefs[key])
        const label = doc.createElement('label')
        label.setAttribute('for', key)
        label.textContent = key.replaceAll('_',' ')
        _(elem).append(label, doc.createElement('br'))
    },
    'bool': (prefs, key, elem) => {
        const ctrl = doc.createElement('input')
        ctrl.setAttribute('type', 'checkbox')
        ctrl.checked = prefs[key]
        ctrl.onchange = () => { prefs[key] = ctrl.checked }
        createCtrl.common(prefs, key, elem, ctrl)
    },
    'num': (prefs, key, elem) => {
        const ctrl = doc.createElement('input')
        const out = doc.createElement('output')
        ctrl.setAttribute('type', 'range')
        ctrl.setAttribute('min', prefs.lo[key])
        ctrl.setAttribute('max', prefs.hi[key])
        ctrl.setAttribute('step', prefs.step[key])
        out.textContent = prefs[key]
        ctrl.oninput = () => {
            prefs[key] = round(parseFloat(ctrl.value),3) 
            out.textContent = prefs[key]
        }
        createCtrl.common(prefs, key, elem, ctrl, out)
    },
    'choice': (prefs, key, elem) => {
        const select = doc.createElement('select')
        for (const opt of prefs.opts[key]) {
            const option = doc.createElement('option')
            option.setAttribute('value', opt)
            option.selected = prefs[key] == opt
            option.textContent = opt
            select.append(option)
        }
        select.onchange = () => { prefs[key] = select.value }
        createCtrl.common(prefs, key, elem, select)
    }    
}

for (const key of phys.keys)
    createCtrl[phys.type[key]](phys, key, '#pedit')   

for (const key of render.keys)
    createCtrl[render.type[key]](render,key,'#rpref')


window.move = false

const handleInput = async (e) => {
    e.preventDefault()
    if (e.type == 'pointerup') {
        move = false
        cv.style.cursor = 'grab'
        sim.dropParticle(e.x, e.y)
    }
    if (e.type == 'pointerdown') {
        move = { x: e.x, y: e.y, btn: e.button }
        if (e.button == 2) {
           sim.grabParticle(e.x, e.y)
           cv.style.cursor = 'grabbing'
        } else if (e.button == 1) {
            cv.style.cursor = 'all-scroll'
        } else if (e.button == 0) {
            cv.style.cursor = 'all-scroll'
        }
    }
    if (e.type == 'pointermove') {
        if (!move) return
        const dx = .005*(e.x - move.x), dy = .005*(move.y - e.y)
        move.x = e.x
        move.y = e.y
        if (move.btn == 0) sim.rotateCam(dx, dy)
        else if (move.btn == 1) sim.strafeCam(dx, dy)
        else if (move.btn == 2) sim.dragParticle(move.x, move.y)
    }
    if (e.type == 'wheel')
        sim.advanceCam(-0.0005 * e.deltaY)
    if (e.type == 'keydown')
        if (e.code == 'Space')
            location.reload()

    return true        
}
    
for (const type of ['pointerup','pointerout','pointerdown','pointermove','contextmenu'])
    cv.addEventListener(type, handleInput, { capture: true, passive: false })
doc.onwheel = handleInput
doc.onkeydown = handleInput
window.onresize = () => {
    cv.width = cv.style.width = window.innerWidth
    cv.height = cv.style.height = window.innerHeight
    sim.resize(cv.width, cv.height)
}

const step = doc.createElement('button')
step.id = 'step'
step.onclick = () => { sim.compute.fwdstep = true }
step.style.display = phys.paused ? 'inline' : 'none'
step.innerHTML = '&#128099;'
phys.watch(['paused'], () => {
    step.style.display = phys.paused ? 'inline' : 'none'
})
const paused = _('input[name="paused"]')
paused.parentNode.insertBefore(step,paused.nextSibling.nextSibling)

async function updateInfo() {
    const lines = [`cam pos: ${sim.camPos.toString()}`]
    let physStat = await sim.compute.stats(), renderStat = await sim.render.stats()
    for (const { kind, fps, profile } of [physStat, renderStat]) {
        lines.push(`${kind} fps:${fps.toFixed(2)}`)
        if (!profile) continue
        for (const [label, nsecs] of profile) lines.push(`${label}: ${nsecs / 1000n} &mu;s`)
        lines.push(`total: ${profile.sum(([label, nsecs]) => nsecs, 0n) / 1000n} &mu;s`)
        lines.push('&nbsp;')
    }
    lines.push(`frameratio(avg): ${(physStat.fps / renderStat.fps).toFixed(3)}`)
    _('#info').innerHTML = lines.join('<br/>')
    setTimeout(updateInfo, 500)
}
setTimeout(updateInfo, 500)

_('#hand').onclick = () => cv.requestPointerLock()
doc.onpointerlockchange = () => {
    sim.activateHand(doc.pointerLockElement == cv)
    doc.onmousemove = doc.onmousemove ? null : (e) => sim.moveHand(e.movementX, e.movementY, 0)
    doc.onwheel = doc.onwheel == handleInput ? (e) => sim.moveHand(0, 0, e.deltaY * .0001) : handleInput
}

sim.run()
