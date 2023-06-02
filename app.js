const { Sim, render, phys, loadWavefront, loadBitmap } = await import('./sim.js')
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

await GeoDB.open()

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
        $(elem).append(html('label', {for: key}, key.replaceAll('_',' ')), html('br'))
    },
    'bool': (prefs, key, elem) => {
        const ctrl = html('input', {type:'checkbox'}).on('change', () => { prefs[key] = ctrl.checked })
        ctrl.checked = prefs[key]
        createCtrl.common(prefs, key, elem, ctrl)
    },
    'num': (prefs, key, elem) => {
        const ctrl = html('output', {}, prefs[key])
        ctrl.on('pointerdown', e => {
            window.on('pointermove', e => {
                let step = prefs.step[key]
                prefs[key] = roundEps(clamp(prefs[key] + e.movementX*step, prefs.lo[key], prefs.hi[key]))
                ctrl.textContent = prefs[key]
            })
            window.on('pointerup', e => {
                window.off(['pointermove','pointerup'])
            })
            e.preventDefault()
        })
        createCtrl.common(prefs, key, elem, ctrl)
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
    if (!render.hidden[key])
        createCtrl[render.type[key]](render,key,'#rpref')

cv.on('pointerdown', down => {
    if (!window.sim) return
    if (down.button == 2) {
       sim.grabParticle(down.x, down.y)
       cv.style.cursor = 'grabbing'
    } else if (down.button == 1 || down.button == 0) {
        cv.style.cursor = 'all-scroll'
    }
    window.on('pointermove', move => {
        const dx = .005*move.movementX, dy = -.005*move.movementY
        if (down.button == 0) sim.rotateCam(dx, dy)
        else if (down.button == 1) sim.strafeCam(dx, dy)
        else if (down.button == 2) sim.moveParticle(move.x, move.y)
    })
    window.on('pointerup', up => {
        cv.style.cursor = 'grab'
        sim.moveParticle(up.x, up.y, true)
        window.off(['pointerup','pointermove'])
    })
    down.preventDefault()
}, { capture: true, passive: false })

cv.on('wheel', wheel => {
    if (!window.sim) return
    sim.advanceCam(-0.001 * wheel.deltaY)
}, { passive: true })

cv.on('contextmenu', menu => menu.preventDefault())

window.on('resize', () => {
    cv.width = cv.style.width = window.innerWidth
    cv.height = cv.style.height = window.innerHeight
    sim.resize(cv.width, cv.height)
})

const step = html('button', {id: 'step'}, '\u{1F463}').on('click', () => { sim.computer.fwdstep() })
step.style.display = phys.paused ? 'inline' : 'none'

const icon = (color,path) =>
      `data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg"%3E%3Cpath style='fill:%23${color}' d='${path}'/%3E%3C/svg%3E`


const playIcon = icon('11DD44', 'M 0,0 32,16 0,32 Z')
const stopIcon = icon('DD1144', 'M 0,0 32,0 32,32 0,32 Z')
$`#icon`.href = phys.paused ? stopIcon : playIcon;

phys.watch(['paused'], () => {
    step.style.display = phys.paused ? 'inline' : 'none'
    $`#icon`.href = phys.paused ? stopIcon : playIcon;
})


render.watch(['color_src','color_dst','alpha_src','alpha_dst'], (k,v) => {
    $('#'+k).value = v
})
const paused = $`input[name="paused"]`
paused.parentNode.insertBefore(step,paused.nextSibling.nextSibling)

async function updateInfo() {
    const lines = []
    let physStat = await sim.computer.stats(), renderStat = await sim.renderer.stats()
    let ttotal = 0n
    for (const { kind, fps, profile } of [physStat, renderStat]) {
        lines.push(`${kind} fps:${fps.toFixed(2)}`)
        if (!profile) continue
        for (const [label, nsecs] of profile) lines.push(`${label}: ${nsecs / 1000n} &mu;s`)
        let steptot = profile.sum(([label, nsecs]) => nsecs, 0n) / 1000n
        if (steptot > 0n) lines.push(`${kind} tot: ${steptot} &mu;s`)
        lines.push('&nbsp;')
        ttotal += steptot * (kind == 'render' ? 1n : BigInt(phys.frameratio))
    }
    lines.push(`frameratio(avg): ${(physStat.fps / renderStat.fps).toFixed(3)}`)
    if (ttotal > 0n) lines.push(`total: ${ttotal} &mu;s`)
    lines.push('&nbsp;')
    lines.push(`cam pos: ${sim.uniforms.cam_pos}`)
    $`#info`.innerHTML = lines.join('<br/>')
    setTimeout(updateInfo, 500)
}



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
        dis.tabButton = html('button',{},name).on('click', () => dis.show())
        return dis
    }
    show() {
        this.style.display = 'flex'
        this.parentNode.active = this
        for (const sibling of this.parentNode.children)
            if (sibling instanceof EditorTable)
                if (sibling != this) {
                    sibling.style.display = 'none'
                    sibling.tabButton.className = ''
                }
        this.update()
        this.tabButton.className = 'opentab'
    }


    async update() {

        const start = this.page * this.nrows
        let results = await transact([this.name],'readonly', async x => await x.query(this.name))
        let ntot = results.size
        results = [...results.entries()].slice(start, start + this.nrows)
        let stop = start + results.length
        this.back.disabled = this.page <= 0
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
                transact(db.storeNames, 'readwrite', async x => {
                    await x.deleteWithRelatives(this.name, id)
                    this.update()
                })
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
                const edit = html('td', { contenteditable: true }, str(orig)).on('blur', () => {
                    let val = edit.textContent
                    if (typeof orig == 'number') val = parseFloat(val)
                    else if (orig instanceof Array) val = eval(val)
                    transact([this.name], 'readwrite', async x => {
                        await x.update(this.name, id, col, val)
                        edit.textContent = str(val)
                    })
                })
                return edit
            })
            const row = html('tr',{},[idcol, ...editcols, delcol])
            return row

        }))
        this.table.append(...rows)
    }
}

let openTransacts = 0
let transact = async (stores, perm, cb) => {
    return await db.transact(stores, perm, async x => {
        if (perm == 'readwrite')
            for (let ctrl of $$('.editor button, .editor input, .editor label')) {
                ctrl.disabled = true
                ctrl.className = 'disabled'
            }
        let result = await cb(x)
        if (perm == 'readwrite') {
            for (let ctrl of $$('.editor button, .editor input, .editor label')) {
                ctrl.disabled = false
                ctrl.className = 'enabled'
            }
            window.editor.active.update()
        }
        return result
    })
}


const stores = {}
const storeBtns = []
for (const name of ['meshes', 'verts', 'faces', 'bitmaps']) {
    const table = new EditorTable(name)
    stores[name] = table
    storeBtns.push(table.tabButton)
}
const storeNav = html('div', { class:'stores' }, storeBtns)
const storeFooter = html('div', { class:'footer' });

[
    ['objfile', '.obj', false, (x,f,d) => loadWavefront(f.name.split('.')[0], d, x)],
    ['pngfile', '.png', true, (x,f,d) => loadBitmap(f.name, d, x)]
].forEach(([id,accept,dataUrl,cb]) =>
    storeFooter.append(html('input', { id, type:'file', accept }).on('change', function() {
        const file = this.files[0]
        if (!file) return
        const reader = new FileReader().on('load', e => {
            transact(db.storeNames, 'readwrite', async x => await cb(x,file, e.target.result))
        })
        if (dataUrl) reader.readAsDataURL(file)
        else reader.readAsText(file)
        this.value = ''
    }), html('label', {for:id, class:'enabled'}, 'Load '+accept)))

storeFooter.append(html('button', {}, 'reset').on('click', async () => {
    await GeoDB.reset()
    window.editor.active.update()
}))
window.editor = html('dialog', { id:'editor', class:'editor' }, [storeNav, ...Object.values(stores), storeFooter])
stores['meshes'].show()
doc.body.append(editor)
$`#scene`.on('click', e => editor.open ? editor.close() : editor.show())

window.sim = await Sim(cv.width, cv.height, ctx)
window.sim.run()

window.debug = (n = 18) => {
    sim.pull('debug').then(data => {
        window.d = new Float32Array(data.buffer)
        dbg({debug:window.d.subarray(0,n).map(x => round(x*1e5)/1e5).join(' ')})
    })
}
setTimeout(updateInfo, 500)




