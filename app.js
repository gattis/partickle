const { Sim, Params, render, phys } = await import('./sim.js')
window.render = render
window.phys = phys
const doc = document

window.$$ = (q,...vals) => {
    if (typeof q != 'string')
        q = q.map((s,i) => s + (i < vals.length ? vals[i] : '')).join('')
    return doc.querySelectorAll(q)
}
window.$ = (...args) => $$(...args)[0]

function html(type, attrs = {}, content = '') {
    const elem = doc.createElement(type)
    if (content instanceof Array) elem.append(...content)
    else elem.append(content)
    for (const [k,v] of Object.entries(attrs))
        elem.setAttribute(k,v)
    return elem
}

window.db = await GeoDB.open()

window.cv = $`canvas`
cv.width = cv.style.width = window.innerWidth
cv.height = cv.style.height = window.innerHeight
cv.style.cursor = 'grab'

window.ctx = cv.getContext('webgpu')

const createCtrl = {
    'common': (prefs,key,elem,ctrl,out) => {
        if (out) $(elem).append(out)
        $(elem).append(ctrl)
        ctrl.setAttribute('id', key)
        ctrl.setAttribute('name', key)
        ctrl.setAttribute('value', prefs[key])
        $(elem).append(html('label', {for: key}, key.replaceAll('_',' ')), doc.createElement('br'))
    },
    'bool': (prefs, key, elem) => {
        const ctrl = html('input', {type:'checkbox'}).on('change', () => { prefs[key] = ctrl.checked })
        ctrl.checked = prefs[key]
        createCtrl.common(prefs, key, elem, ctrl)
    },
    'num': (prefs, key, elem) => {
        const ctrl = html('input', {type: 'range', min: prefs.lo[key], max: prefs.hi[key], step: prefs.step[key]})
        const out = html('output', {}, prefs[key])
        ctrl.on('input', () => {
            prefs[key] = round(parseFloat(ctrl.value),4) 
            out.textContent = prefs[key]
        })
        createCtrl.common(prefs, key, elem, ctrl, out)
    },
    'choice': (prefs, key, elem) => {
        const select = doc.createElement('select').on('change', () => { prefs[key] = select.value })
        for (const opt of prefs.opts[key]) {
            const option = html('option', {value: opt}, opt)
            option.selected = prefs[key] == opt
            select.append(option)
        }
        createCtrl.common(prefs, key, elem, select)
    }    
}

for (const key of phys.keys)
    createCtrl[phys.type[key]](phys, key, '#pedit')   

for (const key of render.keys)
    createCtrl[render.type[key]](render,key,'#rpref')


window.move = false

const handleInput = async (e) => {
    if (e.type != 'wheel') e.preventDefault()
    if (!window.sim) return
    if (e.type == 'pointerup') {
        move = false
        cv.style.cursor = 'grab'
        sim.moveParticle(e.x, e.y, true)
    }
    if (e.type == 'pointerdown') {
        move = { x: e.x, y: e.y, btn: e.button }
        if (e.button == 2) {
            console.log('grab',e.x,e.y)
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
        else if (move.btn == 2) sim.moveParticle(move.x, move.y)
    }
    if (e.type == 'wheel') sim.advanceCam(-0.001 * e.deltaY)

    return true        
}
    
for (const type of ['pointerup','pointerout','pointerdown','pointermove','contextmenu'])
    cv.on(type, handleInput, { capture: true, passive: false })
doc.on('wheel', handleInput, { passive: true})
window.on('resize', () => {
    cv.width = cv.style.width = window.innerWidth
    cv.height = cv.style.height = window.innerHeight
    sim.resize(cv.width, cv.height)
})

const step = html('button', {id: 'step'}, '\u{1F463}').on('click', () => { sim.compute.fwdstep = true })
step.style.display = phys.paused ? 'inline' : 'none'
phys.watch(['paused'], () => {
    step.style.display = phys.paused ? 'inline' : 'none'
})
const paused = $`input[name="paused"]`
paused.parentNode.insertBefore(step,paused.nextSibling.nextSibling)

async function updateInfo() {
    if (sim.stopRequest) return
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
    $`#info`.innerHTML = lines.join('<br/>')
    setTimeout(updateInfo, 500)
}


$`#hand`.on('click', () => cv.requestPointerLock())
doc.on('pointerlockchange', () => {
    let exiting = doc.pointerLockElement != cv
    sim.activateHand(!exiting)
    doc.on('mousemove', exiting ? null : (e) => sim.moveHand(e.movementX, e.movementY, 0))
    doc.on('wheel', exiting ? handleInput : (e) => sim.moveHand(0, 0, e.deltaY * .0001))
})

$`#fix`.on('click', () => sim.fixParticle())


class EditorTable extends HTMLDivElement {
    constructor(name) {
        const dis = Object.setPrototypeOf(html('div'), EditorTable.prototype)
        dis.classList.add('editTable')
        dis.name = name
        dis.page = 0
        dis.nrows = name == 'bitmaps' ? 2 : 25
        dis.counting = html('span')
        dis.back = html('button', {}, '<').on('click', () => { dis.page--; dis.update() })
        dis.fwd = html('button', {}, '>').on('click', () => { dis.page++; dis.update() })
        dis.table = html('table')
        dis.summary = html('div', { class:'summary' }, [html('b', {}, dis.name+': '), dis.back, dis.fwd, dis.counting])
        dis.append(dis.summary, dis.table)
        return dis
    }
    show() {
        this.style.display = 'flex'
        this.parentNode.active = this
        for (const sibling of this.parentNode.children)
            if (sibling instanceof EditorTable) 
                if (sibling != this) 
                    sibling.style.display = 'none'
        this.update()
    }

    async update() {
        this.back.disabled = this.page <= 0
        const start = this.page * this.nrows
        db.transact(this.name)
        let results = await db.query(this.name)
        db.commit()
        let ntot = results.size
        results = [...results.entries()].slice(start, start + this.nrows)
        let stop = start + results.length
        this.fwd.disabled = stop >= ntot
        this.counting.textContent = ` 0 found`
        this.table.replaceChildren()
        if (results.length == 0) return
        
        this.counting.textContent = ` ${start + 1}-${stop} of ${ntot}`
        const cols = Object.keys(results[0][1])
        const rows = [html('tr', {}, [html('th',{},'id'), ...cols.map(col => html('th',{},col))])]
        let str = (val) => {
            if (val == null || val == undefined)  return 'null'
            if (val instanceof Array) return `[${val.map(x => str(x)).join(',')}]`
            return val.toString()
        }
        rows.push(...results.map(([id,result]) => {
            let idcol = html('td',{},id)
            let delcol = html('td', { class:'delrow' }, html('button',{},'\u274C').on('click', async () => {
                await db.deleteWithRelatives(this.name, id)
                this.update()
            }))
            let editcols = cols.map(col => {
                const orig = result[col]
                if (orig instanceof ImageBitmap) {
                    const preview = html('canvas', { width:256, height:256 })
                    ctx = preview.getContext('2d')
                    ctx.scale(256/orig.width, 256/orig.height);
                    ctx.drawImage(orig,0,0);
                    return html('td',{},preview)
                }
                const edit = html('td', { contenteditable: 'true' }, str(orig)).on('blur', () => {
                    let val = edit.textContent
                    if (typeof orig == 'number') val = parseFloat(val)
                    else if (orig instanceof Array) val = eval(val)
                    db.update(this.name, { key:id, col, val });
                    edit.textContent = str(val)
                })
                return edit
            })
            const row = html('tr',{},[idcol, ...editcols, delcol])
            if (this.name == 'meshes')
                row.append(html('button',{},'sample').on('click', () => db.sampleMesh(id, 2*phys.r)))
            return row
            
        }))
        this.table.append(...rows)
    }
}



const stores = {}
const storeBtns = []
for (const name of db.objectStoreNames) {
    const table = new EditorTable(name)
    stores[name] = table
    storeBtns.push(html('button',{},name).on('click', () => table.show()))
}
const storeNav = html('div', { class:'stores' }, storeBtns)       
const storeFooter = html('div', { class:'footer' });

[
    ['objfile', '.obj', false, async (f,d) => await db.loadWavefront(f.name.split('.')[0], d)],
    ['pngfile', '.png', true, async (f,d) => await db.loadBitmap(f.name, d)]
].forEach(([id,accept,dataUrl,cb]) =>
    storeFooter.append(html('input', { id, type:'file', accept }).on('change', function() {
        const file = this.files[0]
        if (!file) return
        this.disabled = true       
        const reader = new FileReader().on('load', async e => { 
            await cb(file, e.target.result)
            editor.active.update()
            this.disabled = false
        })       
        if (dataUrl) reader.readAsDataURL(file)
        else reader.readAsText(file)
        this.value = ''
    }), html('label', {for:id}, 'Load '+accept)))
storeFooter.append(html('button', {}, 'reset').on('click', () => db.reset().then(() => editor.active.update())))
window.editor = html('dialog', { id:'editor', class:'editor' }, [storeNav, ...Object.values(stores), storeFooter])
stores['meshes'].show()
doc.body.append(editor)

window.sim = null
$`#scene`.on('click', e => editor.open ? editor.close() : editor.show())
$`#restart`.on('click', async e => {
    if (window.sim) await sim.stop()
    runSim()
})


const runSim = async () => {
    window.sim = await new Sim().init(cv.width, cv.height, ctx, db)
    window.sim.run()
    setTimeout(updateInfo, 500)
}


runSim()




