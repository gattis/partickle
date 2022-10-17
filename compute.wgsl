type v3 = vec3<f32>;
type v3i = vec3<i32>;
type m3 = mat3x3<f32>;

const MAXNN = ${MAXNN}u;
const REXPAND = 1.07f;

@compute @workgroup_size(${threads})
fn predict(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid:u32 = gid.x;
    if (pid >= arrayLength(&particles)) { return; }       
    let p = &particles[pid];
    (*p).k = 0u;
    (*p).prev_pos = (*p).pos;
    (*p).prev_vel = (*p).vel;
    if (params.grabbing == i32(pid)) { return; }
    let m = &meshes[(*p).mesh];
    if ((*m).flags == 1) { return; }

    (*p).vel *= (1 - 0.1*pow(params.damp, 10.0));

    var agrav = v3(0, 0, -params.gravity * (*m).gravity);
    (*p).vel += agrav * params.t;
    (*p).pos += 0.5 * ((*p).prev_vel + (*p).vel) * params.t;
    
}

@compute @workgroup_size(${threads})
fn cntsort_cnt(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid:u32 = gid.x;
    if (pid >= arrayLength(&particles)) { return; }
    let p = &particles[pid];
    let m = &meshes[(*p).mesh];
    if ((*m).flags == 1) { return; }
    let sd = v3i((*p).pos / (params.r * 2 * REXPAND) + ${threads/2}f);
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
    if ((*m).flags == 1) { return; }
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
    if ((*m).flags == 1) { return; }
    let hash = (*p1).hash;
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
            if ((*p1).mesh == (*p2).mesh && (*m).shape != 0) { continue; }
            if (length((*p1).pos - (*p2).pos) >= (params.r * 2 * REXPAND)) { continue; }
            let k = (*p1).k;
            if (k < MAXNN) {
                (*p1).nn[k] = pid2;
                (*p1).k = k + 1;
            }
        }
    }}}
}


@compute @workgroup_size(${threads})
fn project(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid:u32 = gid.x;
    if (pid >= arrayLength(&particles)) { return; }
    if (params.grabbing == i32(pid)) { return; }
    let p = &particles[pid];
    let m = &meshes[(*p).mesh];
    if ((*m).flags == 1) { return; }

    let vel_in = 0.5 * ((*p).vel + (*p).prev_vel);
    var vel_out = v3(0,0,0);
    var pos_out = v3(0,0,0);
    var cnt = 0.0;
    let mass = (*p).mass * params.density;
    
    let k = min(MAXNN, (*p).k);
    let dsq = 4.0 * params.r * params.r;
    for (var i = 0u; i < k; i += 1) {
        let p2 = &particles[(*p).nn[i]];
        let vel2_in = 0.5 * ((*p2).vel + (*p2).prev_vel);
        let mass2 = (*p2).mass * params.density;
        let dp = (*p).prev_pos - (*p2).prev_pos;
        let dv = vel_in - vel2_in;
        var a = dot(dv,dv);
        let b = 2*dot(dp,dv);
        let c = dot(dp,dp) - dsq;
        var disc = b*b - 4*a*c;
        if (disc < 0) { continue; }
        disc = sqrt(disc);
        a *= 2.0;
        let t1 = (-b - disc)/a;
        let t2 = (-b + disc)/a;
        if (t2 < 0 || t1 > params.t) { continue; }
        let tcol = t1;
        let col_pos = (*p).prev_pos + vel_in * tcol;
        let col2_pos = (*p2).prev_pos + vel2_in * tcol;
        let n = normalize(col_pos - col2_pos);
        let ncom = n * dot(n,vel_in);
        let ncom2 = -n * dot(-n,vel2_in);
        let deltav = (ncom*(mass - mass2) + 2.0*ncom2*mass2)/(mass + mass2) - ncom;
        let vel_new = vel_in + (1-params.collidamp) * deltav;
        let tremain = params.t - tcol;
        pos_out += col_pos + vel_new * tremain;
        vel_out += vel_new;
        cnt += 1.0;
    }
    
    
    if (params.ground > 0 && (*p).pos.z < params.r) {
        let vi = (*p).prev_vel; let vf = (*p).vel; let pi = (*p).prev_pos; let pf = (*p).pos;
        var vo:v3; var po:v3;
        let compliance = 1 - params.collidamp;
        if ((*p).prev_pos.z <= params.r) {
            vo = v3(vf.xy * compliance, 0);
            po = v3(pi.xy + (vi.xy + vo.xy)/2*params.t, params.r);
        } else {
            let a = (vf-vi) / params.t;
            let tcol = (vi.z + sqrt(vi.z*vi.z - 2*a.z*(pi.z - params.r))) / -a.z;
            var vc = (*p).prev_vel + tcol * a;
            let pc = (*p).prev_pos + ((*p).prev_vel + vc)/2 * tcol;
            vc.z = vc.z - compliance * 2 * vc.z;
            let tremain = params.t - tcol;        
            vo = vc + a * tremain; 
            vo = v3(vo.xy * compliance, vo.z);
            po = pc + (vc+vo)/2 * tremain;
        }
        vel_out += vo;
        pos_out += po;
        cnt += 1.0;
    }
    
    if (cnt == 0) {
        (*p).tmp_vel = (*p).vel;
    } else {
        (*p).tmp_vel = vel_out / f32(cnt);
        (*p).pos = pos_out / f32(cnt);
    }
   
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
    if ((*m).flags == 1) { return; }
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
    (*m).ci = centroidwork[0] / (f32(N) + 0.00001);

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
        shapework[gid.x] = mat3x3<f32>(p.x*q, p.y*q, p.z*q);
    }
}

fn mat2Quat(m:mat3x3<f32>) -> vec4<f32> {
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
        if (m[2][2] > m[(u32(i)*4u)/3u][(u32(i)*4u)%3u]) { i = 2; }
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

fn quat2Mat(q:vec4<f32>) -> mat3x3<f32> {
    let qx = 2.0f * q.x * q;
    let qy = 2.0f * q.y * q;
    let qz = 2.0f * q.z * q;
    return mat3x3<f32>(1.0f - qy.y - qz.z,
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
        w = w / wmag;
        let rad = wmag/2;
        let s = sin(rad);
        quat = normalize(quatMul(vec4<f32>(s*w.x, s*w.y, s*w.z, cos(rad)), quat));
    }
    let m = &meshes[0];
    (*m).rot = quat2Mat(quat);

}

type u3 = array<u32,3>;
const tetfaces = array<u3,4>(u3(1,3,2), u3(0,2,3), u3(0,3,1), u3(0,1,2));

@compute @workgroup_size(${threads})
fn constrain(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid = gid.x;
    if (pid >= arrayLength(&particles)) { return; }
    if (params.grabbing == i32(pid)) { return; }    
    let p = &particles[pid];
    let m = &meshes[(*p).mesh];
    if ((*m).flags == 1) { return; }

    (*p).vel = (*p).tmp_vel;

    var dpos_spring = v3(0);
    let nedges = f32((*p).ef - (*p).ei);
    for (var e = (*p).ei; e < (*p).ef; e += 1) {
        let p2 = &particles[edges[e]];
        var delta = (*p2).pos - (*p).pos;
        let dist = length(delta);
        if (dist == 0.0) { continue; }
        let dist0 = length((*p2).rest_pos - (*p).rest_pos);
        dpos_spring += params.edge_stiff * (dist - dist0) / dist * delta / nedges;
    }

    var center:v3;
    var rot:m3;
    if ((*m).flags == 2) {
        center = params.handpos;
        rot = params.handrot;
    } else {
        center = (*m).ci;
        rot = (*m).rot;
    }
    let goal = center + rot * (*p).q;
    var dpos_shape = params.shape_stiff / 100.0 * (goal - (*p).pos);   
    var dpos_tetvol = v3(0);
    let ntets = f32((*p).tf - (*p).ti);
    for (var t = (*p).ti; t < (*p).tf; t += 1) {
        let tet = tets[ptets[t]];
        var k:i32;;
        for (k = 0; k < 4; k += 1) {
            if (tet.verts[k] == pid) { break; }
        }
        if (k == 4) { continue; }

        let a = particles[tet.verts[0]].pos;
        let ab = particles[tet.verts[1]].pos - a;
        let ac = particles[tet.verts[2]].pos - a;
        let ad = particles[tet.verts[3]].pos - a;
        let vol = dot(cross(ab,ac),ad) / 6.0;

        let p0 = particles[tet.verts[tetfaces[k][0]]].pos;
        let p1 = particles[tet.verts[tetfaces[k][1]]].pos;
        let p2 = particles[tet.verts[tetfaces[k][2]]].pos;
        let n = cross(p1 - p0, p2 - p0);

        dpos_tetvol -= params.tetvol_stiff * 300.0 * (vol - tet.vol0) * n / ntets;
    }

    (*p).pos_delta = dpos_spring + dpos_shape + dpos_tetvol;
    (*p).vel += (*p).pos_delta / params.t;

}

@compute @workgroup_size(${threads})
fn syncpos(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid:u32 = gid.x;
    if (pid >= arrayLength(&particles)) { return; }
    if (params.grabbing == i32(pid)) { return; }
    let p = &particles[pid];
    let m = &meshes[(*p).mesh];
    if ((*m).flags == 1) { return; }
    (*p).pos += (*p).pos_delta;
    (*p).pos_delta = v3(0,0,0);
}


