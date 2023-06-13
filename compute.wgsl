alias v3 = vec3<f32>;
alias v4 = vec4<f32>;
alias v3i = vec3<i32>;
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
    var p = pbuf[pid];
    p.prev_pos = p.pos;

    let m = mbuf[p.mesh];

    if (uni.grabbing == i32(pid)) {
        //p.vel = 0.9 * (uni.grabTarget - p.prev_pos) / uni.dt;
        p.pos = uni.grabTarget;
    } else if (p.fixed == 0u) {
        if (p.nedges != 0) {
            let ri = p.pos - m.ci;
            p.vel += min(1.0, uni.dt * 20.0 * uni.damp) * (m.vi + cross(m.wi, ri) - p.vel);
        }
        p.vel.z +=  -uni.gravity * m.gravity * uni.dt;
        p.pos += p.vel * uni.dt;
    } else {
        p.vel = v3(0);
    }

    pbuf[pid] = p;
}



@compute @workgroup_size(${threads})
fn cntsort_cnt(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid:u32 = gid.x;
    if (pid >= arrayLength(&pbuf)) { return; }
    let p = &pbuf[pid];
    var sd = v3i((*p).pos / (2.0 * uni.r) + ${threads/2}f);
    sd.x = min(max(sd.x,0),${threads});
    sd.y = min(max(sd.y,0),${threads});
    sd.z = min(max(sd.z,0),${threads});
    let hash = sd.x + sd.y * ${threads} + sd.z * ${threads**2};
    (*p).hash = hash;
    atomicAdd(&cnts_atomic[hash], 1);
}

@compute @workgroup_size(${threads})
fn prefsum_down(@builtin(local_invocation_id) lid:vec3<u32>,
                @builtin(workgroup_id) wid:vec3<u32>,
                @builtin(num_workgroups) wgs:vec3<u32>) {
    let k = lid.x + wid.x * ${threads};
    for (var stride = 1u; stride < ${threads}u; stride = stride << 1u) {
        let opt = lid.x >= stride;
        let sum = select(0, cnts[k] + cnts[k - stride], opt);
        storageBarrier();
        if (opt) { cnts[k] = sum; }
        storageBarrier();
    }
    if (lid.x != ${threads - 1} || wgs.x == 1) { return; }
    work[wid.x] = cnts[k];
}

@compute @workgroup_size(${threads})
fn prefsum_up(@builtin(local_invocation_id) lid:vec3<u32>, @builtin(workgroup_id) wid:vec3<u32>) {
    cnts[lid.x + (wid.x + 1) * ${threads}] += work[wid.x];
}

@compute @workgroup_size(${threads})
fn cntsort_sort(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid:u32 = gid.x;
    if (pid >= arrayLength(&pbuf)) { return; }
    let p = pbuf[pid];
    let pos = atomicSub(&cnts_atomic[p.hash], 1) - 1;
    sorted[pos] = pid;
}

const incs = array(v3i(-1,0,1), v3i(-1,1,1), v3i(0,-1,1), v3i(0,0,1), v3i(0,1,0), v3i(0,1,1),
                   v3i(1,-1,0), v3i(1,-1,1), v3i(1,0,0), v3i(1,0,1), v3i(1,1,-1), v3i(1,1,0), v3i(1,1,1));

fn collide_pair(pid1:u32, pid2:u32) {
    let p1 = &pbuf[pid1];
    let p2 = &pbuf[pid2];
    let m1 = mbuf[(*p1).mesh];
    let m2 = mbuf[(*p2).mesh];
    if ((*p1).nedges > 0 && (*p2).nedges > 0 && (*p1).mesh == (*p2).mesh) { return; }
    let delta = (*p1).pos - (*p2).pos;
    let dist = length(delta);
    if (dist > 2*uni.r) { return; }
    let dir = select(delta / dist, (*p1).pos, dist == 0);
    let c = 2*uni.r - dist;
    (*p1).pos += uni.collidamp * m1.collidamp * 0.5 * c * dir;
    (*p2).pos -= uni.collidamp * m2.collidamp * 0.5 * c * dir;
}

@compute @workgroup_size(${threads})
fn collide(@builtin(global_invocation_id) gid:vec3<u32>) {
    if (gid.x >= arrayLength(&grid_group)) { return; }
    let hash = i32(grid_group[gid.x]);
    var bin = v3i(0, 0, hash / ${threads**2});
    bin.y = (hash - bin.z * ${threads**2}) / ${threads};
    bin.x = hash - bin.z * ${threads**2} - bin.y * ${threads};
    let istart = cnts[hash];
    let istop = cnts[hash+1];
    
    for (var i = istart; i < istop; i++) {
        let pid1 = sorted[i];
        for (var j = i + 1; j < istop; j++) {
            collide_pair(pid1,sorted[j]);
        }
        for (var b = 0; b < 13; b++) {
            let inc = incs[b];
            let bin2 = v3i(bin.x+inc.x, bin.y+inc.y, bin.z+inc.z);
            if (bin2.x < 0 || bin2.y < 0 || bin2.z < 0) { continue; }
            if (bin2.x >= ${threads} || bin2.y >= ${threads} || bin2.z >= ${threads}) { continue; }
            let hash2 = bin2.x + bin2.y * ${threads} + bin2.z * ${threads**2};
            if (hash2 == hash || hash2 < 0 || hash2 >= ${threads**3}) { continue; }
            let jstop = cnts[hash2 + 1];
            for (var j = cnts[hash2]; j < jstop; j++) {
                collide_pair(pid1, sorted[j]);
            }
        }
    }

}

@compute @workgroup_size(${threads})
fn bounds(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid:u32 = gid.x;
    let npbuf = arrayLength(&pbuf);
    if (pid >= npbuf) { return; }
    var p = pbuf[pid];
    let m = mbuf[p.mesh];
    let collidamp = uni.collidamp * m.collidamp;
    let pmin = -v3(uni.xspace/2, uni.yspace/2, 0) + uni.r;
    let pmax = v3(uni.xspace/2, uni.yspace/2, uni.zspace) - uni.r;
    var cgrad = v3(0);
    let dp = p.pos - p.prev_pos;
                

                   
    for (var axis = 0; axis < 3; axis += 1) {
        if (p.pos[axis] < pmin[axis]) {
            //cgrad[axis] = pmin[axis] - p.pos[axis];
            p.pos[axis] += uni.collidamp * (pmin[axis] - p.pos[axis]);
            //pbuf[pid].prev_pos[axis] = p.pos[axis] + collidamp * dp[axis];
            
        }
        if (p.pos[axis] > pmax[axis]) {
            //cgrad[axis] = pmax[axis] - p.pos[axis];
            p.pos[axis] += uni.collidamp * (pmax[axis] - p.pos[axis]);
            //p.pos[axis] += pmax[axis];
            //pbuf[pid].prev_pos[axis] = p.pos[axis] + collidamp * dp[axis];
        }
    }
    
    


    pbuf[pid].pos = p.pos;
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
    let pstart = pbuf[pidstart];
    let m = mbuf[pstart.mesh];
    var volstiff = uni.volstiff * m.volstiff;
    var shearstiff = uni.shearstiff * m.shearstiff;
    if (volstiff == 0 || shearstiff == 0) { return; }
    shearstiff = 1.0/shearstiff - 1.0;
    volstiff = 0.01 * (1.0/volstiff - 1.0);
           
    var n = pstart.nring;
    if (n < 4) { return; }
    var pids = pstart.rings;
    let s = pstart.s;
    let Qinv = pstart.qinv;

    let c0 = pstart.c0;
    var c = v3();
    var pos:array<v3,64>;
    var R0:array<v3,64>;
    for (var i = 0u; i < n; i++) {
        let randi = (i + uni.seed) % n;
        pos[i] = pbuf[pids[randi]].pos;
        R0[i] = pbuf[pids[randi]].rest_pos - c0;
        c += pos[i];
    }


    c /= f32(n);
    
    var P = m3();
    for (var i = 0u; i < n; i++) {
        let rs = s*(pos[i] - c);
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
        pos[i] += lambda * (G * R0[i]);
        c += pos[i];
    }
    c /= f32(n);

    P = m3();
    for (var i = 0u; i < n; i++) {
        let rs = s*(pos[i] - c);
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
        pos[i] += lambda * (G * R0[i]);
        c += pos[i];
    }
    c /= f32(n);

    P = m3();
    for (var i = 0u; i < n; i++) {
        let rs = s*(pos[i] - c);
        let r0 = R0[i];
        P += m3(rs*r0.x, rs*r0.y, rs*r0.z);
    }

    F = P * Qinv;
    for (var i = 0u; i < n; i++) {
        let randi = (i + uni.seed) % n;
        let pid = pids[randi];
        if (pbuf[pid].fixed == 0) {
            pbuf[pid].pos = c + F * R0[i];
        }
    }
}

@compute @workgroup_size(${threads})
fn update_vel(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid = gid.x;
    if (pid >= arrayLength(&pbuf)) { return; }
    var p = pbuf[pid];
    if (p.fixed == 1) { return; }

    pbuf[pid].vel = (p.pos - p.prev_pos) / uni.dt;
}

@compute @workgroup_size(${threads})
fn avgs_prep(@builtin(global_invocation_id) gid:vec3<u32>) {
    let m = mbuf[0];
    let pstart = m.pi;
    let pstop = m.pf;
    let i = pstart + gid.x;
    if (i < pstop) {
        pavg[gid.x] = pbuf[i].pos;
        vavg[gid.x] = pbuf[i].vel;
        let r = pbuf[i].pos - m.ci;
        let R = m3(0, r.z, -r.y, -r.z, 0, r.x, r.y, -r.x, 0);
        lavg[gid.x] = cross(r, pbuf[i].vel);
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


    
