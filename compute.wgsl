type v3 = vec3<f32>;
type v3i = vec3<i32>;
type m3 = mat3x3<f32>;
type m43 = mat4x3<f32>;
type m2 = mat2x2<f32>;
type u3 = array<u32,3>;

const MAXNN = ${MAXNN}u;
const REXPAND = 1.07f;


@compute @workgroup_size(${threads})
fn predict(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid:u32 = gid.x;
    if (pid >= arrayLength(&particles)) { return; }       
    let p = &particles[pid];
    (*p).k = 0u;
    (*p).prev_pos = (*p).pos;
    (*p).delta_pos = v3(0.0);
    if ((*p).w == 0) { return; }
    let m = &meshes[(*p).mesh];
    if (bool((*m).inactive)) { return; }
    (*p).vel *= 1 - 0.01*pow(params.damp, 4.0);
    var agrav = v3(0, 0, -params.gravity * (*m).gravity);
    (*p).vel += agrav * params.t;
    (*p).pos += (*p).vel * params.t;
}





fn quatMul(a:vec4<f32>, b:vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.x * b.w + a.w * b.x + a.y * b.z - a.z * b.y,
                     a.y * b.w + a.w * b.y + a.z * b.x - a.x * b.z,
                     a.z * b.w + a.w * b.z + a.x * b.y - a.y * b.x,
                     a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

@compute @workgroup_size(${threads})
fn centroid_prep(@builtin(global_invocation_id) gid:vec3<u32>) {
    let m = &meshes[0];
    if (bool((*m).inactive)) { return; }
    let pstart = (*m).pi;
    let pstop = (*m).pf;
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
    let m = &meshes[0];
    (*m).ci = centroidwork[0] * (1.0/(f32(N) + 0.00001));

}

@compute @workgroup_size(${threads})
fn rotate_prep(@builtin(global_invocation_id) gid:vec3<u32>) {
    let m = &meshes[0];
    let pstart = (*m).pi;
    let pstop = (*m).pf;
    let i = pstart + gid.x;
    if (i < pstop) {
        let part = &particles[i];
        let p = (*part).pos - (*m).ci;
        let q = (*part).q;
        //let w = (*part).w;
        shapework[gid.x] = m3(p.x*q, p.y*q, p.z*q);
    }
}

fn mat2Quat(m:m3) -> vec4<f32> {
    let tr = m[0][0] + m[1][1] + m[2][2];
    var out:vec4<f32>;
    if (tr > 0.0) {
        var root = sqrt(tr + 1.0); 
        out.w = 0.5 * root;
        root = 0.5 * (1.0/root);
        out.x = (m[1][2] - m[2][1]) * root;
        out.y = (m[2][0] - m[0][2]) * root;
        out.z = (m[0][1] - m[1][0]) * root;
    } else {
        var i = 0;
        if (m[1][1] > m[0][0]) { i = 1; }
        if (m[2][2] > m[(u32(i)*4u)/3u][(u32(i)*4u)%3u]) { i = 2; }
        var j = (i+1) % 3;
        var k = (i+2) % 3;
        let i4 = i*4; let j4 = j*4; let k4 = k*4;
        var root = sqrt(m[i4/3][i4%3] - m[j4/3][j4%3] - m[k4/3][k4%3] + 1.0);
        out[i] = 0.5 * root;
        root = 0.5 * (1.0/root);
        let j3k=j*3+k; let k3j=k*3+j; let j3i=j*3+i; let i3j=i*3+j; let k3i=k*3+i; let i3k=i*3+k;
        out.w = (m[j3k/3][j3k%3] - m[k3j/3][k3j%3]) * root;
        out[j] = (m[j3i/3][j3i%3] + m[i3j/3][i3j%3]) * root;
        out[k] = (m[k3i/3][k3i%3] + m[i3k/3][i3k%3]) * root;
    }
    return out;
}

fn quat2Mat(q:vec4<f32>) -> m3 {
    let qx = 2.0f * q.x * q;
    let qy = 2.0f * q.y * q;
    let qz = 2.0f * q.z * q;
    return m3(1.0f - qy.y - qz.z,
              qx.y + qz.w,
              qx.z - qy.w,
              qx.y - qz.w,
              1.0f - qx.x - qz.z,
              qy.z + qx.w,
              qx.z + qy.w,
              qy.z - qx.w,
              1.0f - qx.x - qy.y);
    
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

    var A = transpose(shapework[0]);
    var quat = mat2Quat(A);
    
    for (var i = 0; i < 40; i += 1) {
        var R = quat2Mat(quat);
        var w = (cross(R[0],A[0]) + cross(R[1],A[1]) + cross(R[2],A[2])) *
            (1.0 / abs(dot(R[0],A[0]) + dot(R[1],A[1]) + dot(R[2],A[2])) + 1.0e-9);
        let wmag = length(w);
        if (wmag < 1.0e-9) { break; }
        w = w * (1.0 / wmag);
        let rad = wmag * 0.5;
        let s = sin(rad);
        quat = normalize(quatMul(vec4<f32>(s*w.x, s*w.y, s*w.z, cos(rad)), quat));
    }
    let m = &meshes[0];
    (*m).rot = quat2Mat(quat);

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
    if (params.vol_stiff == 0) { return; }
    var stiff = clamp(1-pow(params.vol_stiff, 0.2), 0, 1);
    stiff = stiff / (1.3 - stiff) + 0.005;
    let tid = tetgroup[tidx];
    let pids = tets[tid];
    let pid0 = pids[0];
    var N:m3;
    var F:m3;
    var gs:array<v3,4>;
    var ws:array<f32,4>;
    ws[0] = particles[pid0].w;
    for (var i = 1; i < 4; i++) {
        ws[i] = particles[pids[i]].w;
        N[i - 1] = particles[pids[i]].rest_pos - particles[pid0].rest_pos;
        F[i - 1] = particles[pids[i]].pos - particles[pid0].pos;
    }
    let vol = (1.0 / 6.0) * invert(&N);

    let NT = transpose(N);
    if (tid == 1) { debug[0] = vol; }

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
    
    var walpha = w + stiff / params.t / params.t /  vol;
    var dlambda = select(-c / walpha, 0.0, walpha == 0.0);
    for (var i = 0; i < 4; i++) {
        particles[pids[i]].pos += gs[i] * dlambda * ws[i];
    }
    for (var i = 1; i < 4; i++) {
        F[i - 1] = particles[pids[i]].pos - particles[pid0].pos;
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
        particles[pids[i]].pos += gs[i] * dlambda * ws[i];
    }
}




@compute @workgroup_size(${threads})
fn shapematch(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid = gid.x;
    if (pid >= arrayLength(&particles)) { return; }
    let p = &particles[pid];
    if ((*p).w == 0) { return; }  
    let m = &meshes[(*p).mesh];
    if (bool((*m).inactive)) { return; }
    if (params.shape_stiff == 0) { return; }
    var center:v3;
    var rot:m3;
    if (bool((*m).pose)) {
        center = params.handpos;
        rot = params.handrot;
    } else {
        center = (*m).ci;
        rot = (*m).rot;
    }
    let goal = center + rot * (*p).q;
    (*p).pos += pow(params.shape_stiff,3) * (goal - (*p).pos);  
}


@compute @workgroup_size(${threads})
fn update_vel(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid = gid.x;
    if (pid >= arrayLength(&particles)) { return; }
    let p = &particles[pid];
    if ((*p).w == 0) { return; }  
    let m = &meshes[(*p).mesh];
    if (bool((*m).inactive)) { return; }

    (*p).pos += (*p).delta_pos;
    let delta = (*p).pos - (*p).prev_pos;
    (*p).vel = delta / params.t;


    
}

fn isnan(val:f32) -> bool {
    if (val > 1e20 || val < -1e20) { return true; }
    if (val < 0.0 || 0.0 < val || val == 0.0) { return false; }
    return true;
}

@compute @workgroup_size(${threads})
fn project(@builtin(global_invocation_id) gid:vec3<u32>) {
 
    let pid:u32 = gid.x;
    if (pid >= arrayLength(&particles)) { return; }
    let p = &particles[pid];
    if ((*p).w == 0) { return; }
    let m = &meshes[(*p).mesh];
    if (bool((*m).inactive)) { return; }
    

    let w = max(0.00001, (*p).w);
    
    let k = min(MAXNN, (*p).k);
    let dsq = 4.0 * params.r * params.r;
    for (var i = 0u; i < k; i += 1) {
        let p2 = &particles[(*p).nn[i]];
        let w2 = max(0.00001, (*p2).w);
        
        var grad = (*p).pos - (*p2).pos;
        let dist = length(grad);
        let c = dist - 2.0 * params.r;
        if (c >= 0) { continue; }
        if (dist == 0.0) {
            grad = v3(0,0,1);
        } else {
            grad = grad / dist;
        }
        (*p).delta_pos -= params.collidamp * c * grad;
   }


     if ((*p).pos.z < params.r) {
        //(*p).pos.z = params.r;
        //(*p).vel -= v3((*p).vel.xy * params.collidamp, (2 - params.collidamp) * (*p).vel.z);

        (*p).delta_pos.z += params.collidamp * (params.r - (*p).pos.z);
    }
}




@compute @workgroup_size(${threads})
fn cntsort_cnt(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid:u32 = gid.x;
    if (pid >= arrayLength(&particles)) { return; }
    let p = &particles[pid];
    let m = &meshes[(*p).mesh];
    if (bool((*m).inactive)) { return; }
    let sd = v3i((*p).pos * (1.0 / (params.r * 2 * REXPAND)) + ${threads/2}f);
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
    let p = &particles[pid];
    let m = &meshes[(*p).mesh];
    if (bool((*m).inactive)) { return; }
    let hash = (*p).hash;
    if (hash < 0) { return; }
    let pos = atomicSub(&cnts_atomic[hash], 1) - 1;
    sorted[pos] = pid;
}

@compute @workgroup_size(${threads})
fn grid_collide(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid1:u32 = gid.x;
    if (pid1 >= arrayLength(&particles)) { return; }
    let p1 = &particles[pid1];
    let m = &meshes[(*p1).mesh];
    if (bool((*m).inactive)) { return; }
    let hash = (*p1).hash;
    if (hash < 0) { return; }
    var h = v3i(0, 0, hash / ${threads**2});
    h.y = (hash - h.z * ${threads**2}) / ${threads};
    h.x = hash - h.z * ${threads**2} - h.y * ${threads};
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
            let p2 = &particles[pid2];                    
            if ((*p1).mesh == (*p2).mesh && !bool((*m).fluid)) { continue; }
            if (length((*p1).pos - (*p2).pos) >= (params.r * 2 * REXPAND)) { continue; }
            let k = (*p1).k;
            if (k < MAXNN) {
                (*p1).nn[k] = pid2;
                (*p1).k = k + 1;
            }
        }
    }}}
}

