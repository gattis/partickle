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
@compute @workgroup_size(${threads})
fn predict(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid:u32 = gid.x;
    if (pid >= arrayLength(&particles)) { return; }
    var p = particles[pid];
    p.k = 0u;
    p.prev_pos = p.pos;
    p.delta_pos = v3(0.0);

    let m = meshes[p.mesh];
    if (bool(m.inactive)) { return; }

    if (uniforms.grabbing == i32(pid)) {
        p.vel = 0.3 * (uniforms.grabTarget - p.prev_pos) / uniforms.dt;
    } else if (p.fixed == 0u) {
        let vel = p.vel * (1 - 0.01*pow(uniforms.airdamp, 4.0));
        let velmag = length(vel);
        let veldir = select(vel/velmag, v3(0), velmag == 0);
        p.vel = softmin(velmag, 50, 1.0) * veldir;
        p.vel.z +=  -uniforms.gravity * m.gravity * uniforms.dt;
    } else {
        p.vel = v3(0);
    }
    p.pos += p.vel * uniforms.dt;
    particles[pid] = p;
}

@compute @workgroup_size(${threads})
fn centroid_prep(@builtin(global_invocation_id) gid:vec3<u32>) {
    let m = meshes[0];
    if (bool(m.inactive)) { return; }
    let pstart = m.pi;
    let pstop = m.pf;
    let i = pstart + gid.x;
    if (i < pstop) {
        centroidwork[gid.x] = particles[i].pos;
    }
}

@compute @workgroup_size(${threads})
fn get_centroid(@builtin(local_invocation_id) lid:vec3<u32>,
            @builtin(workgroup_id) wgid:vec3<u32>,
            @builtin(num_workgroups) wgs:vec3<u32>) {
    let N = arrayLength(&centroidwork);
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
    meshes[0].ci = centroidwork[0] * (1.0/(f32(N) + 0.00001));
}

@compute @workgroup_size(${threads})
fn rotate_prep(@builtin(global_invocation_id) gid:vec3<u32>) {
    let m = meshes[0];
    let pstart = m.pi;
    let pstop = m.pf;
    let i = pstart + gid.x;
    if (i < pstop) {
        let part = particles[i];
        let p = part.pos - m.ci;
        let q = part.rest_pos - m.c0;
        shapework[gid.x] = m3(p.x*q, p.y*q, p.z*q);
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

    meshes[0].quat = extract_rotation(transpose(shapework[0]), meshes[0].quat);
}


@compute @workgroup_size(${threads})
fn shapematch(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid = gid.x;
    if (pid >= arrayLength(&particles)) { return; }
    let p = particles[pid];
    if (p.fixed == 1) { return; }
    let m = meshes[p.mesh];
    if (bool(m.inactive)) { return; }
    if (uniforms.shape_stiff == 0) { return; }
    let goal = m.ci + quat2Mat(m.quat) * (p.rest_pos - m.c0);
    particles[pid].pos += pow(uniforms.shape_stiff,3) * (goal - p.pos);
}

@compute @workgroup_size(${threads})
fn normalmatch(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid = gid.x;
    if (pid >= arrayLength(&particles)) { return; }
    let p = particles[pid];
    if (bool(meshes[p.mesh].inactive)) { return; }

    var A = m3(0,0,0,0,0,0,0,0,0);

    for (var i = 0u; i < p.nedges; i++) {
        let posi = particles[p.edges[i]].pos - p.pos;
        let pos0 = particles[p.edges[i]].rest_pos - p.rest_pos;
        A = A + m3(posi.x*pos0, posi.y*pos0, posi.z*pos0);
    }
    let quat = extract_rotation(transpose(A), p.quat);
    particles[pid].quat = quat;
    particles[pid].norm = quat2Mat(quat) * p.norm0;
}


fn invert(m:ptr<function,m3>) -> f32 {
    let p = m3((*m));
    for (var i = 0u; i < 3u; i++) {
        let a = (i + 1u) % 3u;
        let b = (i + 2u) % 3u;
        for (var j = 0u; j < 3u; j++) {
            let c = (j + 1u) % 3u;
            let d = (j + 2u) % 3u;
            (*m)[i][j] = p[a][c]*p[b][d] - p[a][d]*p[b][c];
        }
    }
    let det = dot(p[0], v3((*m)[0][0], (*m)[1][0], (*m)[2][0]));
    (*m) *= 1.0/det;
    return det;
}

@compute @workgroup_size(${threads})
fn neohookean(@builtin(global_invocation_id) gid:vec3<u32>) {
    var tidx = gid.x;
    let ntets = arrayLength(&tetgroup);
    if (tidx >= ntets) { return; }
    if (uniforms.vol_stiff == 0) { return; }
    var stiff = 1.1/(1.1 - uniforms.vol_stiff);
    let tid = tetgroup[tidx];
    let pids = tets[tid];
    let pid0 = pids[0];
    var N:m3;
    var F:m3;
    var ps:array<v3,4>;
    var prev_ps:array<v3,4>;
    var rest_ps:array<v3,4>;
    var gs:array<v3,4>;
    var ws:array<f32,4>;

    let p0 = particles[pid0];
    ps[0] = p0.pos;
    prev_ps[0] = p0.pos;
    rest_ps[0] = p0.rest_pos;
    ws[0] = p0.w;
    for (var i = 1; i < 4; i++) {
        let pi = particles[pids[i]];
        ps[i] = pi.pos;
        prev_ps[i] = pi.pos;
        rest_ps[i] = pi.rest_pos;
        ws[i] = pi.w;
        N[i - 1] = rest_ps[i] - rest_ps[0];
        F[i - 1] = ps[i] - ps[0];
    }
    let vol = (1.0 / 6.0) * invert(&N);

    let NT = transpose(N);

    F *= N;
    var c = 0.0;
    for (var i = 0; i < 3; i++) {
        c += dot(F[i], F[i]);
    }
    c = sqrt(c);
    var G = (1.0 / c) * (F * NT);
    gs[0] = v3(0);
    for (var i = 1; i < 4; i++) {
        gs[i] = G[i - 1];
        gs[0] -= gs[i];
    }
    var w = 0.0;
    for (var i = 0; i < 4; i++) {
        w += ws[i] * dot(gs[i],gs[i]);
    }

    var walpha = w + stiff / uniforms.dt / uniforms.dt /  vol;
    var dlambda = select(-c / walpha, 0.0, walpha == 0.0);
    for (var i = 0; i < 4; i++) {
        ps[i] += gs[i] * dlambda * ws[i];
    }
    for (var i = 1; i < 4; i++) {
        F[i - 1] = ps[i] - ps[0];
    }
    F *= N;
    c = determinant(F) - 1.0;
    G = m3(cross(F[1], F[2]), cross(F[2], F[0]), cross(F[0], F[1])) * NT;
    gs[0] = v3(0);
    for (var i = 1; i < 4; i++) {
        gs[i] = G[i - 1];
        gs[0] -= gs[i];
    }
    w = 0.0;
    for (var i = 0; i < 4; i++) {
        w += ws[i] * dot(gs[i],gs[i]);
    }
    dlambda = select(-c / w, 0.0, w == 0.0);
    for (var i = 0; i < 4; i++) {
        ps[i] += gs[i] * dlambda * ws[i];
    }

    for (var i = 0; i < 4; i++) {
        let delta = ps[i] - prev_ps[i];
        let mag = length(delta);
        if (particles[pids[i]].fixed == 0) {
            particles[pids[i]].pos += delta;
        }
    }


}



fn isnan(val:f32) -> bool {
    if (val > 1e20 || val < -1e20) { return true; }
    if (val < 0.0 || 0.0 < val || val == 0.0) { return false; }
    return true;
}

@compute @workgroup_size(${threads})
fn cntsort_cnt(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid:u32 = gid.x;
    if (pid >= arrayLength(&particles)) { return; }
    let p = &particles[pid];
    if (bool(meshes[(*p).mesh].inactive)) { return; }
    (*p).pos += (*p).delta_pos;
    (*p).delta_pos = v3(0);
    let sd = v3i((*p).pos * (1.0 / (uniforms.r * DEXPAND)) + ${threads/2}f);
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
    if (pid >= arrayLength(&particles)) { return; }
    let p = particles[pid];
    if (bool(meshes[p.mesh].inactive)) { return; }
    if (p.hash < 0) { return; }
    let pos = atomicSub(&cnts_atomic[p.hash], 1) - 1;
    sorted[pos] = pid;
}

@compute @workgroup_size(${threads})
fn find_collisions(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid1:u32 = gid.x;
    if (pid1 >= arrayLength(&particles)) { return; }
    var p1 = particles[pid1];
    if (bool(meshes[p1.mesh].inactive)) { return; }
    let fluid = bool(meshes[p1.mesh].fluid);
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
	    let p2 = particles[pid2];
            if (p1.mesh == p2.mesh && !(p1.nedges == 0 || p2.nedges == 0)) { continue; }
            if (length(p1.pos - p2.pos) >= (uniforms.r * DEXPAND)) { continue; }
            let k = particles[pid1].k;
            if (k < MAXNN) {
                particles[pid1].nn[k] = pid2;
                particles[pid1].k = k + 1;
            } else {
                return;
            }
        }
    }}}

}


@compute @workgroup_size(${threads})
fn collide(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid:u32 = gid.x;
    let nparticles = arrayLength(&particles);
    if (pid >= nparticles) { return; }
    let p = particles[pid];
    if (p.fixed == 1) { return; }
    if (bool(meshes[p.mesh].inactive)) { return; }

    var pos = p.pos;
    let ipos = pos;
    var delta_pos = v3(0);

    let p1fluid = meshes[p.mesh].fluid == 1;
    let k = min(MAXNN, p.k);
    let e = 4.0;
    
    for (var i = 0u; i < k; i += 1) {
        let p2 = particles[p.nn[i]];
	let p2fluid = meshes[p2.mesh].fluid == 1;
        var grad = pos - p2.pos;
        let dist = length(grad);
	if (dist == 0.0) {
            grad = v3(0,0,1);
        } else {
            grad = grad / dist;
        }
	if (p1fluid && p2fluid) {
            if (dist < uniforms.r * DEXPAND) {
                var c = 2 * uniforms.r - dist;
                if (c < 0) {
                    c += pow(c, e) / pow(uniforms.r * (DEXPAND - 2), e - 1.0);
                }
                delta_pos += uniforms.collidamp * c * grad;
           }
	} else {
	    let c = 2.0 * uniforms.r - dist;
	    if (c > 0) { 
                if (!p2fluid && length(p2.norm) > 0) {
                    grad = p2.norm;
	        }
	        delta_pos += uniforms.collidamp * c * grad;
	        let dp = p.prev_pos - pos;
	        let dpt = -cross(cross(dp, grad), grad);
	        delta_pos += dpt * min(1.0, uniforms.dt * 10.0 * uniforms.friction);
            }
	}
    }

    pos += delta_pos;

    let pmin = -v3(uniforms.xspace/2, uniforms.yspace/2, 0) + uniforms.r;
    let pmax = v3(uniforms.xspace/2, uniforms.yspace/2, uniforms.zspace) - uniforms.r;
    var cgrad = v3(0.0);
    let dp = pos - p.prev_pos;
    for (var axis = 0; axis < 3; axis += 1) {
        if (pos[axis] < pmin[axis]) {
            cgrad[axis] = pmin[axis] - pos[axis];
            pos[axis] = pmin[axis];
            particles[pid].prev_pos[axis] = pos[axis] + uniforms.collidamp * dp[axis];
            
        }
        if (pos[axis] > pmax[axis]) {
            cgrad[axis] = pmax[axis] - pos[axis];    
            pos[axis] = pmax[axis];
            particles[pid].prev_pos[axis] = pos[axis] + uniforms.collidamp * dp[axis];
        }
    }
    
    
    let c = length(cgrad);
    let grad = select(v3(0),cgrad/c,c > 0);
    let dpt = -cross(cross(p.prev_pos - pos, grad), grad);
    pos += dpt * min(1.0, uniforms.dt * 10.0 * uniforms.friction);

    
        
    particles[pid].delta_pos = pos - ipos;

}


@compute @workgroup_size(${threads})
fn update_vel(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid = gid.x;
    if (pid >= arrayLength(&particles)) { return; }
    var p = particles[pid];
    if (p.fixed == 1) { return; }
    if (bool(meshes[p.mesh].inactive)) { return; }
    p.pos += p.delta_pos;
    p.delta_pos = v3(0);

    //if (distance(p.pos,p.prev_pos) < 0.00001) {
    //    p.pos = p.prev_pos;
    //}

    let delta = p.pos - p.prev_pos;
    p.vel = delta / uniforms.dt;
    
    particles[pid] = p;

}



