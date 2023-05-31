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
    let kbinv = 1 / kb;
    return a / pow(1.0 + pow(a / b, kb), kbinv);
}

fn isnan(val:f32) -> bool {
    if (val > 1e20 || val < -1e20) { return true; }
    if (val < 0.0 || 0.0 < val || val == 0.0) { return false; }
    return true;
}

fn safenorm(v:v3) -> v3 {
    let l = length(v);
    if (l == 0) { return v; }
    return v / l;
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
    if (bool(m.inactive)) { return; }

    if (uni.grabbing == i32(pid)) {
        p.vel = 0.5 * (uni.grabTarget - p.prev_pos) / uni.dt;
    } else if (p.fixed == 0u) {
        let vel = p.vel * (1 - 0.01*pow(uni.airdamp, 4.0));
        let velmag = length(vel);
        let veldir = select(vel/velmag, v3(0), velmag == 0);
        p.vel = softmin(velmag, 50, 1.0) * veldir;
        //p.vel += uni.airdamp * (m.vi - p.vel);

        p.vel.z +=  -uni.gravity * m.gravity * uni.dt;
    } else {
        p.vel = v3(0);
    }
    p.pos += p.vel * uni.dt;
    pbuf[pid] = p;
}

@compute @workgroup_size(${threads})
fn cntsort_cnt(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid:u32 = gid.x;
    if (pid >= arrayLength(&pbuf)) { return; }
    let p = &pbuf[pid];
    if (bool(mbuf[(*p).mesh].inactive)) { return; }
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
    if (bool(mbuf[p.mesh].inactive)) { return; }
    if (p.hash < 0) { return; }
    let pos = atomicSub(&cnts_atomic[p.hash], 1) - 1;
    sorted[pos] = pid;
}

@compute @workgroup_size(${threads})
fn find_collisions(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid1:u32 = gid.x;
    if (pid1 >= arrayLength(&pbuf)) { return; }
    var p1 = pbuf[pid1];
    if (bool(mbuf[p1.mesh].inactive)) { return; }
    let fluid = bool(mbuf[p1.mesh].fluid);
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
    if (bool(mbuf[p.mesh].inactive)) { return; }

    var pos = p.pos;
    let ipos = pos;
    var delta_pos = v3(0);

    let p1fluid = mbuf[p.mesh].fluid == 1;
    let k = min(MAXNN, p.k);
    let e = 4.0;
    
    for (var i = 0u; i < k; i += 1) {
        let p2 = pbuf[p.nn[i]];
	let p2fluid = mbuf[p2.mesh].fluid == 1;
        var grad = pos - p2.pos;
        let dist = length(grad);
	if (dist == 0.0) {
            grad = v3(0,0,1);
        } else {
            grad = grad / dist;
        }
	if (p1fluid && p2fluid) {
            if (dist < uni.r * DEXPAND) {
                var c = 2 * uni.r - dist;
                if (c < 0) { c += pow(c, e) / pow(uni.r * (DEXPAND - 2), e - 1.0); }
                delta_pos += uni.collidamp * c * grad;
           }
	} else {
            if (dist > 2.0 * uni.r) {
                let c = 2.0 * uni.r - dist;
	        delta_pos += uni.collidamp * c * grad;
	        let dp = p.prev_pos - pos;
	        let dpt = -cross(cross(dp, grad), grad);
	        delta_pos += dpt * min(1.0, uni.dt * 10.0 * uni.friction);
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
            pbuf[pid].prev_pos[axis] = pos[axis] + uni.collidamp * dp[axis];
            
        }
        if (pos[axis] > pmax[axis]) {
            cgrad[axis] = pmax[axis] - pos[axis];    
            pos[axis] = pmax[axis];
            pbuf[pid].prev_pos[axis] = pos[axis] + uni.collidamp * dp[axis];
        }
    }
    
    
    let c = length(cgrad);
    let grad = select(v3(0),cgrad/c,c > 0);
    let dpt = -cross(cross(p.prev_pos - pos, grad), grad);
    pos += dpt * min(1.0, uni.dt * 10.0 * uni.friction);

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

fn trace(m:m3) -> f32 {
    return m[0][0] + m[1][1] + m[2][2];
}


// See "Physically Based Shape Matching" (Macklin et al.)
@compute @workgroup_size(${threads})
fn surfmatch(@builtin(global_invocation_id) gid:vec3<u32>) {

    if (gid.x >= arrayLength(&frags)) { return; }
    let frag = frags[gid.x];
    let m = mbuf[pbuf[fbuf[frag.start]].mesh];    
    if (bool(m.inactive)) { return; }
    if (frag.stop <= frag.start) { return; }

    if (uni.surf_stiff == 0) { return; }
    let stiff = uni.surf_stiff;
    
    let n = frag.stop - frag.start;
    var pos:array<v3,${maxFragSize}>;
    var pos0:array<v3,${maxFragSize}>;
  
    var c = v3(0);
    var c0 = v3(0);
    for (var i = 0u; i < n; i++) {
        let pid = fbuf[i+frag.start];
        pos[i] = pbuf[pid].pos;
        pos0[i] = pbuf[pid].rest_pos;
        c += pos[i];
        c0 += pos0[i];
    }
    c /= f32(n);
    c0 /= f32(n);

    var P = m3(0,0,0,0,0,0,0,0,0);
    var Q = m3(0,0,0,0,0,0,0,0,0);
    for (var i = 0u; i < n; i++) {
        let r = pos[i] - c;
        let r0 = pos0[i] - c0;
        P += m3(r*r0.x, r*r0.y, r*r0.z);
        Q += m3(r0*r0.x, r0*r0.y, r0*r0.z);
    }
    let Qinv = invert(Q);
    var F = P * Qinv;
    var C = sqrt(dot(F[0],F[0]) + dot(F[1],F[1]) + dot(F[2],F[2]));
    if (C == 0) { return; }
    
    var G = 1.0/C * F * transpose(Qinv);
    var walpha = stiff / uni.dt / uni.dt;
    for (var i = 0u; i < n; i++) {
        let grad = G * (pos0[i] - c0);
        walpha += dot(grad,grad);
    }
    c = v3(0);
    var lambda = select(-C / walpha, 0.0, walpha == 0.0);
    for (var i = 0u; i < n; i++) {
        pos[i] += lambda * (G * (pos0[i] - c0));
        c += pos[i];
    }
    c /= f32(n);

    P = m3(0,0,0,0,0,0,0,0,0);
    for (var i = 0u; i < n; i++) {
        let r = pos[i] - c;
        let r0 = pos0[i] - c0;
        P += m3(r*r0.x, r*r0.y, r*r0.z);
    }
    F = P * Qinv;
    C = determinant(F) - 1.0;

    G = m3(cross(F[1], F[2]), cross(F[2], F[0]), cross(F[0], F[1])) * transpose(Qinv);
    walpha = 0.0;
    for (var i = 0u; i < n; i++) {
        let grad = G * (pos0[i] - c0);
        walpha += dot(grad,grad);
    }

    c = v3(0);
    lambda = select(-C / walpha, 0.0, walpha == 0.0);
    for (var i = 0u; i < n; i++) {
        pos[i] += lambda * (G * (pos0[i] - c0));
        c += pos[i];
    }
    c /= f32(n);

    P = m3(0,0,0,0,0,0,0,0,0);
    for (var i = 0u; i < n; i++) {
        let r = pos[i] - c;
        let r0 = pos0[i] - c0;
        P += m3(r*r0.x, r*r0.y, r*r0.z);
    }
    F = P * Qinv;

    for (var i = 0u; i < n; i++) {
        let f = i + frag.start;
        let pid = fbuf[f];
        if (f < frag.aux && pbuf[pid].fixed == 0) {
            pbuf[pid].delta_pos = c + F * (pos0[i] - c0) - pbuf[pid].pos;
        }
    }
}



@compute @workgroup_size(${threads})
fn centroid_prep(@builtin(global_invocation_id) gid:vec3<u32>) {
    let m = mbuf[0];
    if (bool(m.inactive)) { return; }
    let pstart = m.pi;
    let pstop = m.pf;
    let i = pstart + gid.x;
    if (i < pstop) {
        centroidwork[gid.x] = pbuf[i].pos;
    }
}


@compute @workgroup_size(${threads})
fn get_centroid(@builtin(local_invocation_id) lid:vec3<u32>,
            @builtin(workgroup_id) wgid:vec3<u32>,
            @builtin(num_workgroups) wgs:vec3<u32>) {
    let N = arrayLength(&centroidwork);
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
            centroidwork[stride*oi] += centroidwork[stride*oj];
        }
        storageBarrier();
    }
    if (wgs.x != 1 || lid.x != 0) { return; }
    
    mbuf[0].ci = centroidwork[0] / f32(N);
}

@compute @workgroup_size(${threads})
fn rotate_prep(@builtin(global_invocation_id) gid:vec3<u32>) {
    let m = mbuf[0];
    let pstart = m.pi;
    let pstop = m.pf;
    let i = pstart + gid.x;
    if (i < pstop) {
        let part = pbuf[i];
        let p = part.pos - m.ci;
        let q = part.rest_pos - m.c0;
        shapework[gid.x] = m3(p*q.x, p*q.y, p*q.z);
    }
}

fn quat2Mat(q:v4) -> m3 {
    let qx = 2.0f * q.x * q;
    let qy = 2.0f * q.y * q;
    let qz = 2.0f * q.z * q;
    return m3(1.0f - qy.y - qz.z, qx.y + qz.w, qx.z - qy.w,
              qx.y - qz.w, 1.0f - qx.x - qz.z, qy.z + qx.w,
              qx.z + qy.w, qy.z - qx.w, 1.0f - qx.x - qy.y);
}

fn quatMul(a:v4, b:v4) -> v4 {
    return v4(a.x * b.w + a.w * b.x + a.y * b.z - a.z * b.y,
                     a.y * b.w + a.w * b.y + a.z * b.x - a.x * b.z,
                     a.z * b.w + a.w * b.z + a.x * b.y - a.y * b.x,
                     a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

fn extract_rotation(A:m3, q0:v4) -> v4 {
    var quat = q0;
    for (var i = 0; i < 40; i += 1) {
        var R = quat2Mat(quat);
	let z = abs(dot(R[0],A[0]) + dot(R[1],A[1]) + dot(R[2],A[2])) + 0.00001;
        var w = (cross(R[0],A[0]) + cross(R[1],A[1]) + cross(R[2],A[2])) / z;
        let wmag = length(w);
	if (wmag < 1e-9) { break; }
        w /= wmag;
        let rad = wmag * 0.5;
        let s = sin(rad);
	quat = quatMul(v4(s*w.x, s*w.y, s*w.z, cos(rad)), quat);
	quat = quat / (length(quat) + 0.00001);
    }
    return quat;
}

@compute @workgroup_size(${threads})
fn get_rotate(@builtin(local_invocation_id) lid:vec3<u32>,
              @builtin(workgroup_id) wgid:vec3<u32>,
              @builtin(num_workgroups) wgs:vec3<u32>) {
    let N = arrayLength(&shapework);
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
            shapework[stride*oi] += shapework[stride*oj];
        }
        storageBarrier();
    }
    if (wgs.x != 1 || lid.x != 0) { return; }

    mbuf[0].quat = extract_rotation(shapework[0], mbuf[0].quat);
}

@compute @workgroup_size(${threads})
fn shapematch(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid = gid.x;
    if (pid >= arrayLength(&pbuf)) { return; }
    let p = pbuf[pid];
    if (p.fixed == 1) { return; }
    let m = mbuf[p.mesh];
    if (bool(m.inactive)) { return; }
    if (uni.shape_stiff == 0) { return; }
    let goal = m.ci + quat2Mat(m.quat) * (p.rest_pos - m.c0);
    pbuf[pid].pos += pow(uni.shape_stiff,3) * (goal - p.pos);
}

@compute @workgroup_size(${threads})
fn update_vel(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid = gid.x;
    if (pid >= arrayLength(&pbuf)) { return; }
    var p = pbuf[pid];
    if (p.fixed == 1) { return; }
    if (bool(mbuf[p.mesh].inactive)) { return; }

    //if (distance(p.pos,p.prev_pos) < 0.00001) {
    //    p.pos = p.prev_pos;
    //}

    let delta = p.pos - p.prev_pos;
    p.vel = delta / uni.dt;    
    pbuf[pid] = p;

}


@compute @workgroup_size(${threads})
fn vavg_prep(@builtin(global_invocation_id) gid:vec3<u32>) {
    let m = mbuf[0];
    if (bool(m.inactive)) { return; }
    let pstart = m.pi;
    let pstop = m.pf;
    let i = pstart + gid.x;
    if (i < pstop) {
        vavgwork[gid.x] = pbuf[i].vel;
    }
}


@compute @workgroup_size(${threads})
fn get_vavg(@builtin(local_invocation_id) lid:vec3<u32>,
            @builtin(workgroup_id) wgid:vec3<u32>,
            @builtin(num_workgroups) wgs:vec3<u32>) {
    let N = arrayLength(&vavgwork);
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
            vavgwork[stride*oi] += vavgwork[stride*oj];
        }
        storageBarrier();
    }
    if (wgs.x != 1 || lid.x != 0) { return; }
    
    mbuf[0].vi = vavgwork[0] / f32(N);
}


