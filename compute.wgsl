alias v3 = vec3<f32>;
alias v4 = vec4<f32>;
alias v3i = vec3<i32>;
alias m3 = mat3x3<f32>;
alias m43 = mat4x3<f32>;
alias m2 = mat2x2<f32>;
alias u3 = array<u32,3>;

const MAXNN = ${MAXNN}u;
const DEXPAND = 2.3f;

fn softmin(a:f32, b:f32, k:f32) -> f32 {
    let kb = k*b;
    return a / pow(1.0 + pow(a / b, kb), 1.0/kb);
}

@compute @workgroup_size(${threads})
fn predict(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid:u32 = gid.x;
    if (pid >= arrayLength(&pbuf)) { return; }
    var p = pbuf[pid];
    p.k = 0u;
    p.prev_pos = p.pos;
    p.delta_pos = v3(0.0);

    let m = mbuf[p.mesh];

    if (uni.grabbing == i32(pid)) {
        p.pos = uni.grabTarget;
    } else if (p.fixed == 0u) {
        let ri = p.pos - m.ci;
        p.vel += min(1.0, uni.dt * 20.0 * uni.damp) * (m.vi + cross(m.wi, ri) - p.vel);
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
    let sd = v3i((*p).pos * (1.0 / (uni.r * DEXPAND)) + ${threads/2}f);
    if (sd.x < 0 || sd.y < 0 || sd.z < 0 || sd.x >= ${threads} || sd.y >= ${threads} || sd.z >= ${threads}) {
        (*p).hash = -1;
        return;
    }
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
    if (p.hash < 0) { return; }
    let pos = atomicSub(&cnts_atomic[p.hash], 1) - 1;
    sorted[pos] = pid;
}

@compute @workgroup_size(${threads})
fn find_collisions(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid1:u32 = gid.x;
    if (pid1 >= arrayLength(&pbuf)) { return; }
    var p1 = pbuf[pid1];
    if (p1.hash < 0) { return; }
    var h = v3i(0, 0, p1.hash / ${threads**2});
    h.y = (p1.hash - h.z * ${threads**2}) / ${threads};
    h.x = p1.hash - h.z * ${threads**2} - h.y * ${threads};
    let hstart = max(h - 1, v3i(0,0,0));
    let hstop = min(h + 1, v3i(${threads-1}));

    for (var x = hstart.x; x <= hstop.x; x += 1) {
    for (var y = hstart.y; y <= hstop.y; y += 1) {
    for (var z = hstart.z; z <= hstop.z; z += 1) {
        let ohash = x + y * ${threads} + z * ${threads**2};
        if (ohash < 0 || ohash >= ${threads**3 - 1}) { continue; }
        let ostop = cnts[ohash + 1];
        for (var i = cnts[ohash]; i < ostop; i = i + 1) {
            let pid2 = sorted[i];
            if (pid2 == pid1) { continue; }
	    let p2 = pbuf[pid2];
            if (p1.nedges > 0 && p2.nedges > 0 && p1.mesh == p2.mesh) { continue; }

            if (length(p1.pos - p2.pos) >= (uni.r * DEXPAND)) { continue; }
            let k = pbuf[pid1].k;
            if (k < MAXNN) {
                pbuf[pid1].nn[k] = pid2;
                pbuf[pid1].k = k + 1;
            } else {
                return;
            }
        }
    }}}

}


@compute @workgroup_size(${threads})
fn collide(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid:u32 = gid.x;
    let npbuf = arrayLength(&pbuf);
    if (pid >= npbuf) { return; }
    let p = pbuf[pid];
    if (p.fixed == 1) { return; }

    let m = mbuf[p.mesh];
    let collidamp = uni.collidamp * m.collidamp;
    let friction = uni.friction * m.friction;
    
    var pos = p.pos;
    let ipos = pos;
    var delta_pos = v3(0);
       
    let k = min(MAXNN, p.k);
    let e = 4.0;
    
    for (var i = 0u; i < k; i += 1) {
        let p2 = pbuf[p.nn[i]];
        let m2 = mbuf[p2.mesh];
        var grad = pos - p2.pos;
        let dist = length(grad);
	if (dist == 0.0) {
            grad = v3(0,0,1);
        } else {
            grad = grad / dist;
        }
	if (m.fluid == 1 && m2.fluid == 1) {
            if (dist < uni.r * DEXPAND) {
                var c = 2 * uni.r - dist;
                if (c < 0) { c += pow(c, e) / pow(uni.r * (DEXPAND - 2), e - 1.0); }
                delta_pos += collidamp * m2.collidamp * c * grad;
           }
	} else {
            if (dist > 2.0 * uni.r) {
                let c = 2.0 * uni.r - dist;
	        delta_pos += collidamp * m2.collidamp * c * grad;
	        let dp = p.prev_pos - pos;
	        let dpt = -cross(cross(dp, grad), grad);
	        delta_pos += dpt * min(1.0, uni.dt * 20.0 * friction * m2.friction);
            }
	}
    }

    pos += delta_pos;

    let pmin = -v3(uni.xspace/2, uni.yspace/2, 0) + uni.r;
    let pmax = v3(uni.xspace/2, uni.yspace/2, uni.zspace) - uni.r;
    var cgrad = v3(0.0);
    let dp = pos - p.prev_pos;
    for (var axis = 0; axis < 3; axis += 1) {
        if (pos[axis] < pmin[axis]) {
            cgrad[axis] = pmin[axis] - pos[axis];
            pos[axis] = pmin[axis];
            pbuf[pid].prev_pos[axis] = pos[axis] + collidamp * dp[axis];
            
        }
        if (pos[axis] > pmax[axis]) {
            cgrad[axis] = pmax[axis] - pos[axis];    
            pos[axis] = pmax[axis];
            pbuf[pid].prev_pos[axis] = pos[axis] + collidamp * dp[axis];
        }
    }
    
    
    let c = length(cgrad);
    let grad = select(v3(0),cgrad/c,c > 0);
    let dpt = -cross(cross(p.prev_pos - pos, grad), grad);
    pos += dpt * min(1.0, uni.dt * 20.0 * friction);

    pbuf[pid].delta_pos = pos - ipos;

}

@compute @workgroup_size(${threads})
fn update_pos(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid:u32 = gid.x;
    if (pid >= arrayLength(&pbuf)) { return; }
    pbuf[pid].pos += pbuf[pid].delta_pos;
    pbuf[pid].delta_pos = v3(0);
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
    inv *= 1.0/det;
    return inv;
}

// Derived from "Physically Based Shape Matching" (Muller et al.)
@compute @workgroup_size(${threads})
fn surfmatch(@builtin(global_invocation_id) gid:vec3<u32>) {

    if (gid.x >= arrayLength(&group)) { return; }
    let pidstart = group[gid.x];
    let pstart = pbuf[pidstart];
    let m = mbuf[pstart.mesh];
    var volstiff = uni.volstiff * m.volstiff;
    var shearstiff = uni.shearstiff * m.shearstiff;
    if (volstiff == 0 || shearstiff == 0) { return; }
    shearstiff = 1.0/shearstiff - 1.0;
    volstiff = 0.01 * (1.0/volstiff - 1.0);
    
    var n = pstart.nring;
    var pids = pstart.rings;
    let s = pstart.s;
    let Qinv = pstart.qinv;

    let c0 = pstart.c0;
    var c = v3(0);
    var pos:array<v3,64>;
    var pos0:array<v3,64>;
    for (var i = 0u; i < n; i++) {
        pos[i] = pbuf[pids[i]].pos;
        pos0[i] = pbuf[pids[i]].rest_pos;
        c += pos[i];
    }
    
    c /= f32(n);
    
    var P = m3(0,0,0,0,0,0,0,0,0);
    for (var i = 0u; i < n; i++) {
        let r = pos[i] - c;
        let r0 = pos0[i] - c0;
        P += m3(s*r*r0.x, s*r*r0.y, s*r*r0.z);
    }

    var F = P * Qinv;
    var C = sqrt(dot(F[0],F[0]) + dot(F[1],F[1]) + dot(F[2],F[2]));
    if (C == 0) { return; }
    
    var G = 1.0/C * (F * transpose(Qinv));
    var walpha = shearstiff / uni.dt / uni.dt;
    for (var i = 0u; i < n; i++) {
        let grad = G * (s*(pos0[i] - c0));
        walpha += dot(grad,grad);
    }

    c = v3(0);
    var lambda = select(-C / walpha, 0.0, walpha == 0.0);
    for (var i = 0u; i < n; i++) {
        pos[i] += lambda * (G * (s*(pos0[i] - c0)));
        c += pos[i];
    }
    c /= f32(n);

    P = m3(0,0,0,0,0,0,0,0,0);
    for (var i = 0u; i < n; i++) {
        let r = pos[i] - c;
        let r0 = pos0[i] - c0;
        P += m3(s*r*r0.x, s*r*r0.y, s*r*r0.z);
    }

    F = P * Qinv;
    C = determinant(F) - 1.0;

    G = m3(cross(F[1],F[2]),cross(F[2],F[0]),cross(F[0],F[1])) * transpose(Qinv);
    walpha = volstiff / uni.dt / uni.dt;
    for (var i = 0u; i < n; i++) {
        let grad = G * (s*(pos0[i] - c0));
        walpha += dot(grad,grad);
    }

    c = v3(0);
    lambda = select(-C / walpha, 0.0, walpha == 0.0);
    for (var i = 0u; i < n; i++) {
        pos[i] += lambda * (G * (s*(pos0[i] - c0)));
        c += pos[i];
    }
    c /= f32(n);

    P = m3(0,0,0,0,0,0,0,0,0);
    for (var i = 0u; i < n; i++) {
        let r = pos[i] - c;
        let r0 = pos0[i] - c0;
        P += m3(s*r*r0.x, s*r*r0.y, s*r*r0.z);
    }
    F = P * Qinv;
    for (var i = 0u; i < n; i++) {
        let pid = pids[i];
        if (pbuf[pid].fixed == 0) {
            pbuf[pid].pos = c + F * (pos0[i] - c0);
            dbuf[pid] = dbuf[pid] + 1;
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


    
