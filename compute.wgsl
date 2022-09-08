type v3 = vec3<f32>;
type m3 = mat3x3<f32>;
type v3i = vec3<i32>;

const MAXNN = ${MAXNN}u;
const D = ${D}f;
const Dplus = ${D*1.5};
const T = ${T}f;
const v3zero = v3(0.0);
const m3zero = m3(0f,0f,0f,0f,0f,0f,0f,0f,0f);

@compute @workgroup_size(${threads})
fn predict(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid:u32 = gid.x;
    if (pid >= arrayLength(&particles)) { return; }
    let p = &particles[pid];
    (*p).k = 0u;
    (*p).sp = (*p).si;
    
    let m = &meshes[(*p).mesh];
    var fext = (*m).fext;
    fext = fext - params.friction * (*p).v;
 
    (*p).v = (*p).v + fext * T;
    (*p).sp += (*p).v * T;

    

    
}

@compute @workgroup_size(${threads})
fn cntsort_cnt(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid:u32 = gid.x;
    if (pid >= arrayLength(&particles)) { return; }
    
    let sd = v3i(particles[pid].sp / Dplus + ${threads/2}f);
    if (sd.x < 0 || sd.y < 0 || sd.z < 0 || sd.x >= ${threads} || sd.y >= ${threads} || sd.z >= ${threads}) {
        particles[pid].hash = -1;
        return;
    }
    let hash = sd.x + sd.y * ${threads} + sd.z * ${threads**2};
    particles[pid].hash = hash;
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
    let hash = particles[pid].hash;
    if (hash < 0) { return; }
    let pos = atomicSub(&cnts_atomic[hash], 1) - 1;
    sorted[pos] = pid;
}

@compute @workgroup_size(${threads})
fn grid_collide(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid1:u32 = gid.x;
    if (pid1 >= arrayLength(&particles)) { return; }

    let p1 = &particles[pid1];
    let hash = (*p1).hash;
    var h = v3i(0, 0, hash / ${threads**2});
    h.y = (hash - h.z * ${threads**2}) / ${threads};
    h.x = hash - h.z * ${threads**2} - h.y * ${threads};
    let hstart = max(h - 1, v3i(0,0,0));
    let hstop = min(h + 1, v3i(${threads-1}));

    for (var x = hstart.x; x <= hstop.x; x++) {
    for (var y = hstart.y; y <= hstop.y; y++) {
    for (var z = hstart.z; z <= hstop.z; z++) {
        let ohash = x + y * ${threads} + z * ${threads**2};
        if (ohash < 0 || ohash >= ${threads**3 - 1}) { continue; }
        let ostop = cnts[ohash + 1];
        for (var i = cnts[ohash]; i < ostop; i = i + 1) {
            let pid2 = sorted[i];
            if (pid2 == pid1) { continue; }
            let p2 = &particles[pid2];                    
            if ((*p1).mesh == (*p2).mesh) { continue; }
            if (length((*p1).sp - (*p2).sp) >= Dplus) { continue; }
            let k = (*p1).k;
            if (k < MAXNN) {
                (*p1).nn[k] = pid2;
                (*p1).k = k + 1;
            }
        }
    }}}    
}


@compute @workgroup_size(${threads})
fn collisions(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid:u32 = gid.x;
    if (pid >= arrayLength(&particles)) { return; }
    let p = &particles[pid];

    let k = min(MAXNN, (*p).k);
    var ds = v3(0.0, 0.0, 0.0);
    for (var i = 0u; i < k; i++) {
        let p2 = &particles[(*p).nn[i]];
        let d = length((*p).si - (*p2).si);
        let c = max(0.0, D - d);
        ds -= params.fcol * c * (*p2).grad;
        
    }  
    if (k > 0) {
        ds = ds / f32(k);
        (*p).si = (*p).si + ds;
        (*p).sp = (*p).sp + ds;
    }
}


fn quat2Mat(q:vec4<f32>) -> m3 {
    let qx = 2 * q.x * q;
    let qy = 2 * q.y * q;
    let qz = 2 * q.z * q;
    return m3(1 - qy.y - qz.z,
              qx.y + qz.w,
              qx.z - qy.w,
              qx.y - qz.w,
              1 - qx.x - qz.z,
              qy.z + qx.w,
              qx.z + qy.w,
              qy.z - qx.w,
              1 - qx.x - qy.y);
    
}

fn mat2Quat(m:m3) -> vec4<f32> {
    let tr = m[0][0] + m[1][1] + m[2][2];
    var out:vec4<f32>;
    if (tr > 0.0) {
        var root = sqrt(tr + 1.0); 
        out.w = 0.5 * root;
        root = 0.5 / root;
        out.x = (m[1][2] - m[2][1]) * root;
        out.y = (m[2][0] - m[0][2]) * root;
        out.z = (m[0][1] - m[1][0]) * root;
    } else {
        var i = 0;
        if (m[1][1] > m[0][0]) { i = 1; }
        if (m[2][2] > m[(i*4)/3][(i*4)%3]) { i = 2; }
        var j = (i+1) % 3;
        var k = (i+2) % 3;
        let i4 = i*4; let j4 = j*4; let k4 = k*4;
        var root = sqrt(m[i4/3][i4%3] - m[j4/3][j4%3] - m[k4/3][k4%3] + 1.0);
        out[i] = 0.5 * root;
        root = 0.5 / root;
        let j3k=j*3+k; let k3j=k*3+j; let j3i=j*3+i; let i3j=i*3+j; let k3i=k*3+i; let i3k=i*3+k;
        out.w = (m[j3k/3][j3k%3] - m[k3j/3][k3j%3]) * root;
        out[j] = (m[j3i/3][j3i%3] + m[i3j/3][i3j%3]) * root;
        out[k] = (m[k3i/3][k3i%3] + m[i3k/3][i3k%3]) * root;
    }
    return out;
}

fn quatMul(a:vec4<f32>, b:vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.x * b.w + a.w * b.x + a.y * b.z - a.z * b.y,
                     a.y * b.w + a.w * b.y + a.z * b.x - a.x * b.z,
                     a.z * b.w + a.w * b.z + a.x * b.y - a.y * b.x,
                     a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

@compute @workgroup_size(${threads})
fn centroid_init(@builtin(global_invocation_id) gid:vec3<u32>) {
    let m = &meshes[0];
    let pstart = (*m).pi;
    let pstop = (*m).pf;
    let i = pstart + gid.x;
    if (i < pstop) {
        centroidwork[gid.x] = particles[i].sp;
    }
}

@compute @workgroup_size(${threads})
fn centroid(@builtin(local_invocation_id) lid:vec3<u32>,
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
    (*m).ci = centroidwork[0] / (f32(N) + 0.00001);

}

@compute @workgroup_size(${threads})
fn shapematch_init(@builtin(global_invocation_id) gid:vec3<u32>) {
    let m = &meshes[0];
    let pstart = (*m).pi;
    let pstop = (*m).pf;
    let i = pstart + gid.x;
    if (i < pstop) {
        let part = &particles[i];
        let p = (*part).sp - (*m).ci;
        let q = (*part).q;
        shapework[gid.x] = m3(p.x*q, p.y*q, p.z*q);
    }
}

@compute @workgroup_size(${threads})
fn shapematch(@builtin(local_invocation_id) lid:vec3<u32>,
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
    
    for (var i = 0; i < 35; i++) {
        var R = quat2Mat(quat);
        var w = (cross(R[0],A[0]) + cross(R[1],A[1]) + cross(R[2],A[2])) *
            (1.0 / abs(dot(R[0],A[0]) + dot(R[1],A[1]) + dot(R[2],A[2])) + 1.0e-9);
        let wmag = length(w);
        if (wmag < 1.0e-9) { break; }
        w = w / wmag;
        let rad = wmag/2;
        let s = sin(rad);
        quat = normalize(quatMul(vec4<f32>(s*w.x, s*w.y, s*w.z, cos(rad)), quat));
    }
    let m = &meshes[0];
    (*m).rot = quat2Mat(quat);

}
    
@compute @workgroup_size(${threads})
fn project(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid:u32 = gid.x;
    let nparticles = arrayLength(&particles);
    if (pid >= nparticles) { return; }
    let p = &particles[pid];
    let m = &meshes[(*p).mesh];  
    
    var sf = (*p).sp;    
    let k = min(MAXNN, (*p).k);
    for (var i = 0u; i < k; i++) {
        let p2 = &particles[(*p).nn[i]];
        let d = length((*p).sp - (*p2).sp);
        let c = max(0.0, D - d);
        sf -= params.fcol * c * (*p2).grad;
    }


    let goal = (*m).ci + (*m).rot*(*p).q;
    sf += params.fshape * (*m).fshape * (goal - (*p).sp);
    sf += (*p).lock * ((*p).s0 - (*p).sp);

    let v = (sf - (*p).si) / T;
 
    (*p).v = v;
    (*p).si = sf;
    
}

@compute @workgroup_size(${threads})
fn vertpos(@builtin(global_invocation_id) gid:vec3<u32>) {
    let vid:u32 = gid.x;
    let nverts = arrayLength(&vertices);
    if (vid >= nverts) { return; }
    let v = &vertices[vid];
    let pid = (*v).particle;
    if (pid >= 0) {
        (*v).pos = particles[pid].si;
    } else {
        let m = &meshes[(*v).mesh];
        (*v).pos = (*m).rot * (*v).q + (*m).ci;
    }
}

@compute @workgroup_size(${threads})
fn sort_tris(@builtin(global_invocation_id) gid:vec3<u32>) {
    let tid:u32 = gid.x;
    let ntris = arrayLength(&tris);
    if (tid >= ntris) { return; }
    let tri = &tris[tid];
    for (var i = 0; i < 3; i += 1) {
        let tv = &(*tri)[i];
        let v = &vertices[(*tv).vidx];
        (*tv).pos = (*v).pos;
        (*tv).norm = (*v).norm;
        (*tv).mesh = (*v).mesh;
    }
}


@compute @workgroup_size(${threads})
fn normals(@builtin(global_invocation_id) gid:vec3<u32>) {
    let vid:u32 = gid.x;
    let nverts = arrayLength(&vertices);
    if (vid >= nverts) { return; }
    let v = &vertices[vid];
    let nedges = (*v).nedges;
    let pos = (*v).pos;
    var norm = v3(0f,0f,0f);
    for (var i = 0u; i < nedges; i = i + 1u) {
        let b = vertices[(*v).edges[i % nedges]].pos - pos;
        let c = vertices[(*v).edges[(i+1u) % nedges]].pos - pos;
        let n = cross(c,b);
        let area = 0.5 * length(n);
        let angle = acos(dot(normalize(b),normalize(c)));
        norm += angle * normalize(n);
    }
    norm = normalize(norm);
    (*v).norm = norm;
    let pid = (*v).particle;
    if (pid >= 0) {
        particles[pid].grad = norm;
    }
}

