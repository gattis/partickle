alias v3 = vec3<f32>;
alias v4 = vec4<f32>;
alias v3i = vec3<i32>;
alias v3u = vec3<u32>;
alias m3 = mat3x3<f32>;
alias m43 = mat4x3<f32>;
alias m2 = mat2x2<f32>;
alias u3 = array<u32,3>;

fn softmin(a:f32, b:f32, k:f32) -> f32 {
    let kb = k*b;
    return a / pow(1.0 + pow(a / b, kb), 1.0/kb);
}
    

@compute @workgroup_size(${threads})
fn predict(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid:u32 = gid.x;
    if (pid >= arrayLength(&pbuf)) { return; }
    let p = &pbuf[pid]; 

    
    /*
      if ((*p).nedges != 0) {
      let m = mbuf[(*p).mesh];
      let ri = (*p).x - m.ci;
      (*p).v += min(1.0, uni.dt * 20.0 * uni.damp) * (m.vi + cross(m.wi, ri) - (*p).v);
      }*/

    (*p).v += uni.a * uni.dt;
        
    (*p).xprev = (*p).x;
    (*p).x += (*p).v * uni.dt;

    if (uni.grabStart == 1) {
        let co = uni.cam_x - (*p).x;
        let b = dot(uni.grabRay, co);
        let discrim = b*b - dot(co,co) + uni.r*uni.r;
        if (discrim >= 0) {
            let dist = -b - sqrt(discrim);
            if (dist > 0) {
                let spot = atomicAdd(&hitlist.len, 1);
                if (spot < 1024) {
                    hitlist.list[spot].pid = pid;
                    hitlist.list[spot].x = (*p).x;
                }
            }
        }
    }
   
    if (uni.grabbing == i32(pid)) {
        (*p).grab = 1;
        (*p).x = uni.grabTarget;        
    } else if ((*p).grab == 1) {
        (*p).grab = 0;
        (*p).x += .1*(uni.grabTarget - (*p).xprev);
    } else if ((*p).fixed == 1) {
        (*p).v = v3(0);
        (*p).x = (*p).xprev;
    }
    
    (*p).pmin = (*p).x;
    (*p).pmax = (*p).x;

}


const rcell = 1f;

@compute @workgroup_size(${threads})
fn find_bounds(@builtin(local_invocation_id) lid:vec3<u32>,
               @builtin(workgroup_id) wgid:vec3<u32>,
               @builtin(num_workgroups) wgs:vec3<u32>) {
    let N = arrayLength(&pbuf);
    var n = N;
    var stride = 1u;
    let wgs1 = (n + ${threads} - 1) / ${threads};
    if (wgs.x != wgs1) {
        n = wgs1;
        stride = ${threads}u;
        let wgs2 = (n + ${threads} - 1) / ${threads};
        if (wgs.x != wgs2) {
            n = wgs2;
            stride = ${threads**2}u;
        }
    }
    let offset = wgid.x * ${threads};
    let i = lid.x;
    let oi = offset + i;
    var inc = 1u;
    for (; inc < ${threads}u ;) {
        let j = i + inc;
        let oj = offset + j;
        inc = inc << 1u;
        if (i%inc == 0 && j < ${threads}u && oj < n) {         
            pbuf[stride*oi].pmin = min(pbuf[stride*oi].pmin, pbuf[stride*oj].pmin);
            pbuf[stride*oi].pmax = max(pbuf[stride*oi].pmax, pbuf[stride*oj].pmax);
        }
        storageBarrier();
    }
    if (wgs.x != 1 || lid.x != 0) { return; }

    bounds.min = pbuf[0].pmin - uni.r;
    bounds.max = pbuf[0].pmax + uni.r;
    let span = bounds.max - bounds.min;
    bounds.grid = v3i(max(v3(3,3,3),ceil(span / (2 * uni.r * rcell))));
    bounds.stride = v3i(1, bounds.grid.x, bounds.grid.x * bounds.grid.y);
    bounds.grid.z = min(bounds.grid.z, ${NHASH} / bounds.stride.z);    
    for (var cset = 0; cset < 27; cset++) {
        atomicStore(&setlens[cset][0], 0u);
        atomicStore(&setlens[cset][1], u32(cset));       
    }
    atomicStore(&ncells, 0u);

}


@compute @workgroup_size(${threads})
fn cellcount(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid:u32 = gid.x;
    if (pid >= arrayLength(&pbuf)) { return; }
    let p = &pbuf[pid];
    var cell = v3i(((*p).x - bounds.min) / (2 * uni.r * rcell));
    cell = (bounds.grid + cell % bounds.grid) % bounds.grid;    
    let hash = dot(cell, bounds.stride);
    (*p).cellpos = atomicAdd(&cnts[hash], 1);
    (*p).hash = hash;
}

@compute @workgroup_size(${threads})
fn initcells(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid:u32 = gid.x;
    if (pid >= arrayLength(&pbuf)) { return; }
    let p = &pbuf[pid];
    if ((*p).cellpos != 0) { return; }
    let hash = (*p).hash;
    let cid = atomicAdd(&ncells, 1u);
    cells[cid].npids = u32(min(${CELLCAP-1}, atomicLoad(&cnts[hash])));
    atomicStore(&cnts[hash], i32(cid) + 1);
    let cellv = (hash / bounds.stride) % bounds.grid;
    let cset = dot(cellv % 3, v3i(1,3,9));
    var setpos = atomicAdd(&setlens[cset][0], 1u);
    cellsets[cset][setpos] = i32(cid);
}

const incs = array<v3i,13>(v3i(-1,0,1), v3i(-1,1,1), v3i(0,-1,1), v3i(0,0,1), v3i(0,1,0), v3i(0,1,1),
                   v3i(1,-1,0), v3i(1,-1,1), v3i(1,0,0), v3i(1,0,1), v3i(1,1,-1), v3i(1,1,0), v3i(1,1,1));

@compute @workgroup_size(${threads})
fn fillcells(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid:u32 = gid.x;
    if (pid >= arrayLength(&pbuf)) { return; }
    if (pid == 0) {
        for (var cset = 0; cset < 27; cset++) {
            setdps[cset] = v3u(u32(ceil(f32(atomicLoad(&setlens[cset][0])))), 1, 1);
            idp = v3u(u32(ceil(f32(atomicLoad(&ncells))/${threads}.)), 1, 1);
        }
    }
    let p = &pbuf[pid];
    let hash = (*p).hash;
    let cid = atomicLoad(&cnts[hash]) - 1;
    if ((*p).cellpos >= ${CELLCAP}) { return; }
    cells[cid].pids[(*p).cellpos] = pid;
    if ((*p).cellpos != 0) { return; }
    let cellv = (hash / bounds.stride) % bounds.grid;    
    for (var i = 0; i < 13; i++) {
        var vadj = cellv + incs[i];
        vadj = (bounds.grid + vadj % bounds.grid) % bounds.grid;                
        let adjhash = dot(vadj, bounds.stride);
        cells[cid].adj[i] = atomicLoad(&cnts[adjhash]) - 1;
    }
    
}

struct CollideParam { fluid:bool, ec:f32, ef:f32, mesh:u32, iw:f32 };
fn collide_param(pid:u32) -> CollideParam {
    let p = &pbuf[pid];
    return CollideParam((*p).nedges > 0, uni.collision*(*p).collision, uni.friction*(*p).friction,
                        (*p).mesh, select(1.,0.,(*p).grab == 1 || (*p).fixed == 1));    
}


fn collide_pair(ipid:u32, jpid:u32, m:CollideParam) {
    let ip = &pbuf[ipid];
    let jp = &pbuf[jpid];
    if (m.fluid && m.mesh == (*jp).mesh) { return; }
    let r = (*ip).x - (*jp).x;
    let dist = length(r);
    if (dist >= 2*uni.r) { return; }
    let n = select(r / dist, v3(0,0,1), dist == 0);
    let jw = select(1.,0., (*jp).grab == 1 || (*jp).fixed == 1);
    let w = m.iw + jw;
    if (w == 0) { return; }
    let v = (*ip).v - (*jp).v;
    let vt = v - n*dot(v,n);
    let dx = 2*m.ec*(*jp).collision*(2*uni.r - dist)*n - pow(m.ef*(*jp).friction, 3)*vt*uni.dt;
    (*ip).x += m.iw/w * dx;
    (*jp).x -= jw/w * dx;    
}




@compute @workgroup_size(${threads})
fn intercell(@builtin(global_invocation_id) gid:vec3<u32>) {
    let cid = gid.x;
    if (cid >= atomicLoad(&ncells)) { return; }
    var cell = cells[cid];
    for (var i = 0u; i < cell.npids; i++) {
        let randi = (i + uni.seed) % cell.npids;
        let pid = cell.pids[randi];
        let m = collide_param(pid);
        for (var j = randi+1; j < cell.npids; j++) {
            collide_pair(pid, cell.pids[j], m);
        }

    }
}


@compute @workgroup_size(1)
fn intracell(@builtin(global_invocation_id) gid:vec3<u32>) {
    if (gid.x >= setlen[0]) { return; }

    let cid = u32(cellset[gid.x]);
    var icell = cells[cid];
    for (var i = 0u; i < icell.npids; i++) {
        let randi = (i + uni.seed) % icell.npids;
        let ipid = icell.pids[randi];
        let m = collide_param(ipid);
        for (var a = 0; a < 13; a++) {
            let jcid = icell.adj[a];
            if (jcid == -1) { continue; }
            let jcell = cells[jcid];
            for (var j = 0u; j < jcell.npids; j++) {
                let randj = (j + uni.seed) % jcell.npids;
                let jpid = jcell.pids[randj];
                collide_pair(ipid, jpid, m);
            }
        }
    }
}

                                                            
const plane_normals = array<v3,6>(v3(1,0,0),v3(0,1,0),v3(0,0,1),v3(-1,0,0),v3(0,-1,0),v3(0,0,-1));

@compute @workgroup_size(${threads})
fn collide_bounds(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid = gid.x;
    if (pid >= arrayLength(&pbuf)) { return; }
    let p = &pbuf[pid];
    if ((*p).fixed == 1 || (*p).grab == 1) { return; }

    let ec = uni.collision * (*p).collision;
    let ef = pow(uni.friction * (*p).friction, 3.0);
    for (var i = 0; i < 6; i++) {
        let n = plane_normals[i];
        let point = select(uni.spacemin, uni.spacemax, i >= 3);
        let dist = uni.r - dot((*p).x - point, n);
        if (dist < 0) { continue; }
        let v = (*p).v;
        let vt = v - n*dot(v,n);       
        (*p).x += 2*ec*dist*n - ef*vt*uni.dt;
    }    
}

fn invert(m:m3) -> m3 {
    var inv = m3();
    for (var i = 0u; i < 3u; i++) {
        let a = (i + 1u) % 3u;
        let b = (i + 2u) % 3u;
        for (var j = 0u; j < 3u; j++) {
            let c = (j + 1u) % 3u;
            let d = (j + 2u) % 3u;
            inv[i][j] = m[a][c]*m[b][d] - m[a][d]*m[b][c];
        }
    }
    let det = dot(m[0], v3(inv[0][0], inv[1][0], inv[2][0]));
    if (det != 0) { inv *= 1.0/det; }
    return inv;
}

// Derived from "Physically Based Shape Matching" (Muller et al.)
@compute @workgroup_size(${threads})
fn surfmatch(@builtin(global_invocation_id) gid:vec3<u32>) {
    let ngroup = arrayLength(&group);
    if (gid.x >= ngroup) { return; }
    var pidstart = group[gid.x];
    let pstart = &pbuf[pidstart];
    let m = mbuf[(*pstart).mesh];
    var volstiff = uni.volstiff * m.volstiff;
    var shearstiff = uni.shearstiff * m.shearstiff;
    if (volstiff == 0 || shearstiff == 0) { return; }
    shearstiff = 1.0/shearstiff - 1.0;
    volstiff = 0.01 * (1.0/volstiff - 1.0);
           
    var n = (*pstart).nring;
    if (n < 4) { return; }
    var pids = (*pstart).rings;
    let s = (*pstart).s;
    let Qinv = (*pstart).qinv;

    let c0 = (*pstart).c0;
    var c = v3();
    var x:array<v3,64>;
    var R0:array<v3,64>;
    for (var i = 0u; i < n; i++) {
        let randi = (i + uni.seed) % n;
        x[i] = pbuf[pids[randi]].x;
        R0[i] = pbuf[pids[randi]].x0 - c0;
        c += x[i];
    }


    c /= f32(n);
    
    var P = m3();
    for (var i = 0u; i < n; i++) {
        let rs = s*(x[i] - c);
        let r0 = R0[i];
        P += m3(rs*r0.x, rs*r0.y, rs*r0.z);
    }

    var F = P * Qinv;
    var C = sqrt(dot(F[0],F[0]) + dot(F[1],F[1]) + dot(F[2],F[2]));
    if (C == 0) { return; }
    
    var G = s/C * (F * transpose(Qinv));
    var walpha = shearstiff / uni.dt / uni.dt;
    for (var i = 0u; i < n; i++) {
        let grad = G * R0[i];
        walpha += dot(grad,grad);
    }

    c = v3();
    var lambda = select(-C / walpha, 0.0, walpha == 0.0);
    for (var i = 0u; i < n; i++) {
        x[i] += lambda * (G * R0[i]);
        c += x[i];
    }
    c /= f32(n);

    P = m3();
    for (var i = 0u; i < n; i++) {
        let rs = s*(x[i] - c);
        let r0 = R0[i];
        P += m3(rs*r0.x, rs*r0.y, rs*r0.z);
    }
    F = P * Qinv;
    C = determinant(F) - 1.0;

    G = s * m3(cross(F[1],F[2]),cross(F[2],F[0]),cross(F[0],F[1])) * transpose(Qinv);
    walpha = volstiff / uni.dt / uni.dt;
    for (var i = 0u; i < n; i++) {
        let grad = G * R0[i];
        walpha += dot(grad,grad);
    }

    c = v3();
    lambda = select(-C / walpha, 0.0, walpha == 0.0);
    for (var i = 0u; i < n; i++) {
        x[i] += lambda * (G * R0[i]);
        c += x[i];
    }
    c /= f32(n);

    P = m3();
    for (var i = 0u; i < n; i++) {
        let rs = s*(x[i] - c);
        let r0 = R0[i];
        P += m3(rs*r0.x, rs*r0.y, rs*r0.z);
    }

    F = P * Qinv;
    for (var i = 0u; i < n; i++) {
        let randi = (i + uni.seed) % n;
        let pid = pids[randi];
        if (pbuf[pid].fixed != 1 && pbuf[pid].grab != 1) {
            pbuf[pid].x = c + F * R0[i];
        }
    }
}

@compute @workgroup_size(${threads})
fn update_vel(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid = gid.x;
    if (pid >= arrayLength(&pbuf)) { return; }
    let p = &pbuf[pid];
    if ((*p).fixed == 1 || (*p).grab == 1) {
        (*p).v = v3(0);
        return;
    }
    (*p).v = ((*p).x - (*p).xprev) / uni.dt;
}

@compute @workgroup_size(${threads})
fn avgs_prep(@builtin(global_invocation_id) gid:vec3<u32>) {
    let m = mbuf[0];
    let pstart = m.pi;
    let pstop = m.pf;
    let i = pstart + gid.x;
    if (i < pstop) {
        pavg[gid.x] = pbuf[i].x;
        vavg[gid.x] = pbuf[i].v;
        let r = pbuf[i].x - m.ci;
        let R = m3(0, r.z, -r.y, -r.z, 0, r.x, r.y, -r.x, 0);
        lavg[gid.x] = cross(r, pbuf[i].v);
        iavg[gid.x] = R*transpose(R);
    }
}


@compute @workgroup_size(${threads})
fn avgs_calc(@builtin(local_invocation_id) lid:vec3<u32>,
            @builtin(workgroup_id) wgid:vec3<u32>,
            @builtin(num_workgroups) wgs:vec3<u32>) {
    let N = arrayLength(&vavg);
    if (N == 0) { return; }
    var n = N;
    var stride = 1u;
    let wgs1 = (n + ${threads} - 1) / ${threads};
    if (wgs.x != wgs1) {
        n = wgs1;
        stride = ${threads}u;
        let wgs2 = (n + ${threads} - 1) / ${threads};
        if (wgs.x != wgs2) {
            n = wgs2;
            stride = ${threads**2}u;
        }
    }
    let offset = wgid.x * ${threads};
    let i = lid.x;
    let oi = offset + i;
    var inc = 1u;
    for (; inc < ${threads}u ;) {
        let j = i + inc;
        let oj = offset + j;
        inc = inc << 1u;
        if (i%inc == 0 && j < ${threads}u && oj < n) {         
            pavg[stride*oi] += pavg[stride*oj];
            vavg[stride*oi] += vavg[stride*oj];
            lavg[stride*oi] += lavg[stride*oj];
            iavg[stride*oi] += iavg[stride*oj];
        }
        storageBarrier();
    }
    if (wgs.x != 1 || lid.x != 0) { return; }

    mbuf[0].ci = pavg[0] / f32(N);
    mbuf[0].vi = vavg[0] / f32(N);
    mbuf[0].wi = invert(iavg[0]) * lavg[0];

}


    
