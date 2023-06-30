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
    var p = pbuf[pid]; 

    let m = mbuf[p.mesh];
    /*if ((*p).nedges != 0) {
      let ri = (*p).x - m.ci;
      (*p).v += min(1.0, uni.dt * 20.0 * uni.damp) * (m.vi + cross(m.wi, ri) - (*p).v);
      }*/

    p.v += v3(0,0,-uni.gravity * m.gravity * uni.dt);
    p.xprev = p.x;
    p.x += p.v * uni.dt;

    if (uni.grabStart == 1) {
        let co = uni.cam_x - p.x;
        let b = dot(uni.grabRay, co);
        let discrim = b*b - dot(co,co) + uni.r*uni.r;
        if (discrim >= 0) {
            let dist = -b - sqrt(discrim);
            if (dist > 0) {
                let spot = atomicAdd(&hitlist.len, 1);
                if (spot < 1024) {
                    hitlist.list[spot].pid = pid;
                    hitlist.list[spot].x = p.x;
                }
            }
        }
    }
   
    if (uni.grabbing == i32(pid)) {
        p.grab = 1;
        //        p.xprev = uni.grabTarget;
        p.x = uni.grabTarget;
        
    } else if (p.grab == 1) {
        p.grab = 0;
        p.x += .1*(uni.grabTarget - p.xprev);
    }

    if (p.fixed == 1) {
        p.v = v3(0);
        p.x = p.xprev;
    }
    
    p.pmin = p.x;
    p.pmax = p.x;
    p.dv = v3(0);
    
    pbuf[pid] = p;

}


const rbin = 1f;

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

    var b:Bounds;
    b.min = pbuf[0].pmin;
    b.max = pbuf[0].pmax;
    let span = b.max - b.min;
    b.bins = v3i(floor(span / (2 * uni.r * rbin)) + 1);
    b.stride = v3i(1, b.bins.x, b.bins.x * b.bins.y);
    b.nbins = b.bins.x * b.bins.y * b.bins.z;
    b.chunks = b.bins / 3 + v3i(1,1,1);
    b.chunk_stride = v3i(1, b.chunks.x, b.chunks.x * b.chunks.y);
    b.nchunks = b.chunks.x * b.chunks.y * b.chunks.z;
    b.dispatch = v3u(u32(ceil(f32(b.nchunks) / ${threads}f)), 1, 1);
    bounds_wr = b;
}



@compute @workgroup_size(${threads})
fn cntsort_cnt(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid:u32 = gid.x;
    if (pid >= arrayLength(&pbuf)) { return; }
    let p = &pbuf[pid];
    var sd = v3i(((*p).x - bounds.min) / (2 * uni.r * rbin));
    sd = min(max(sd, v3i(0)), bounds.bins - 1);
    let hash = dot(sd,bounds.stride);
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
    let x = atomicSub(&cnts_atomic[p.hash], 1) - 1;
    sorted[x] = pid;
}

@compute @workgroup_size(${threads})
fn collide(@builtin(global_invocation_id) gid:vec3<u32>) {
    var cid = i32(gid.x);
    let zchunk = cid / bounds.chunk_stride.z;
    cid -= zchunk * bounds.chunk_stride.z;
    let ychunk = cid / bounds.chunk_stride.y;
    let xchunk = cid - ychunk * bounds.chunk_stride.y;
    let bin = 3 * v3i(xchunk,ychunk,zchunk) + off;
    if (any(bin < v3i(0)) || any(bin >= bounds.bins)) { return; }
    let hash = dot(bin,bounds.stride); 
    if (hash < 0 || hash >= bounds.nbins) { return; }
    let incs = array(v3i(-1,0,1), v3i(-1,1,1), v3i(0,-1,1), v3i(0,0,1), v3i(0,1,0), v3i(0,1,1),
                     v3i(1,-1,0), v3i(1,-1,1), v3i(1,0,0), v3i(1,0,1), v3i(1,1,-1), v3i(1,1,0), v3i(1,1,1));
    let istart = cnts[hash];
    let istop = cnts[hash + 1];    
    for (var i = istart; i < istop; i++) {
        let pi = &pbuf[sorted[i]];
        for (var b = 0; b <= 13; b++) {
            var jstart = i + 1;
            var jstop = istop;
            if (b < 13) {
                let bin2 = bin + incs[b];
                if (any(bin2 < v3i(0)) || any(bin2 >= bounds.bins)) { continue; }
                let hash2 = dot(bin2,bounds.stride);
                if (hash2 == hash || hash2 < 0 || hash2 >= bounds.nbins) { continue; }
                jstart = cnts[hash2];
                jstop = cnts[hash2 + 1];
            }
            for (var j = jstart; j < jstop; j++) {
                let pj = &pbuf[sorted[j]];
                if ((*pi).nedges > 0 && (*pj).nedges > 0 && (*pi).mesh == (*pj).mesh) { continue; }
                var xi = (*pi).x; var xj = (*pj).x;
                let r = xi - xj;
                let dist = length(r);
                let c = 2*uni.r - dist;
                if (c <= 0) { continue; }
                let n = select(r / dist, v3(0,0,1), dist == 0);
                let wi = select(1.0, 0.0, (*pi).grab == 1 || (*pi).fixed == 1);
                let wj = select(1.0, 0.0, (*pj).grab == 1 || (*pj).fixed == 1);
                let w = wi + wj;
                if (w == 0) { continue; }              
                xi += wi/w * c * n;
                xj -= wj/w * c * n;               
                let v1 = (xi - (*pi).xprev)/uni.dt - (xj - (*pj).xprev)/uni.dt;
                let v0 = (*pi).v - (*pj).v;
                let v1n = dot(v1,n)*n;
                let e = uni.collision * (*pi).collision * (*pj).collision;
                var dv = -(v1n + n * min(e * dot(v0,n), 0));
                let v1t = v1 - v1n;
                dv -= v1t * pow(uni.friction * (*pi).friction * (*pj).friction, 3);
                (*pi).x = xi;
                (*pj).x = xj;
                (*pi).dv += wi/w*dv;
                (*pj).dv -= wj/w*dv;                
            }
        }
    }
}
                                                            
    

@compute @workgroup_size(${threads})
fn collide_bounds(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid = gid.x;
    if (pid >= arrayLength(&pbuf)) { return; }
    var p = pbuf[pid];
    if (p.fixed == 1 || p.grab == 1) { return; }

    let plane_normals = array(v3(1,0,0),v3(0,1,0),v3(0,0,1),v3(-1,0,0),v3(0,-1,0),v3(0,0,-1));
    let plane_points = array(uni.spacemin, uni.spacemin, uni.spacemin, uni.spacemax, uni.spacemax, uni.spacemax);
    for (var i = 0; i < 6; i++) {
        let n = plane_normals[i];
        let dist = uni.r - dot(p.x - plane_points[i], n);
        if (dist < 0) { continue; }
        p.x += dist*n;
        let v = (p.x - p.xprev)/uni.dt;
        let vn = n*dot(v,n);
        let e = uni.collision * p.collision;
        p.dv -= vn + n * min(e * dot(p.v,n), 0);
        let vt = v - vn;
        p.dv -= vt * pow(uni.friction * p.friction, 3);
    }    
    pbuf[pid] = p;
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
    var p = pbuf[pid];
    if (p.fixed == 1 || p.grab == 1) {
        pbuf[pid].v = v3(0);
        return;
    }
    p.v = (p.x - p.xprev) / uni.dt + p.dv;    
    pbuf[pid] = p;  
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


    
