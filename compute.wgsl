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

fn cell_insert(hash:i32, pid:u32) {
    let pids = &cells[hash];
    for (var i = 0; i < ${CELLSIZE}; i++) {       
        let res = atomicCompareExchangeWeak(&((*pids)[i]), -1, i32(pid));
        if (res.exchanged) { return; }
    }
}

fn cell_delete(hash:i32, pid:u32) {
    let pids = &cells[hash];
    for (var i = 0; i < ${CELLSIZE}; i++) {
        let res = atomicCompareExchangeWeak(&((*pids)[i]), i32(pid), -1);
        if (res.exchanged) { return; }
    }
}    

const cell_stride = v3i(1, ${CELLDIM}, ${CELLDIM**2});
fn to_hash(x: v3) -> i32 {
    return dot((v3i(x / (2*u.r)) % ${CELLDIM} + ${CELLDIM}) % ${CELLDIM}, cell_stride);
}



@compute @workgroup_size(${threads})
fn predict(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid:u32 = gid.x;
    if (pid >= arrayLength(&pbuf)) { return; }
    let p = &pbuf[pid];     
    
    if (u.grab_query == 1) {
        let co = u.cam_x - (*p).x;
        let b = dot(u.grab_ray, co);
        let discrim = b*b - dot(co,co) + u.r*u.r;
        if (discrim >= 0) {
            let dist = -b - sqrt(discrim);
            if (dist > 0) {
                let spot = atomicAdd(&grab_hits.len, 1);
                if (spot < 1024) {
                    grab_hits.list[spot] = GrabHit(pid, (*p).x);
                }
            }
        }
    }   

    if (u.grab_pid == i32(pid)) { // grabbed
        (*p).vprev = (*p).v;
        (*p).v = v3(0);
        (*p).xprev = u.grab_x;
        (*p).x = u.grab_x;
        (*p).w = 0;
    } else if ((*p).w == 0 && w0[pid] != 0) { // dropped
        (*p).vprev = (*p).v;
        (*p).v = (u.grab_x - (*p).x) / u.dt;
        (*p).xprev = (*p).x;
        (*p).x = u.grab_x;
        (*p).w = w0[pid];
    } else if (w0[pid] == 0) { // fixed
        (*p).vprev = (*p).v;
        (*p).v = v3(0);
        (*p).xprev = (*p).x;
        (*p).w = 0;
    } else { // usual
        (*p).vprev = ((*p).x - (*p).xprev) / u.dt;
        (*p).v = (*p).vprev + u.a * u.dt;
        (*p).xprev = (*p).x;
        (*p).x += ((*p).vprev + (*p).v)/2 * u.dt;
        (*p).w = w0[pid];
    }

    /*
      if ((*p).nedges != 0) {
      let m = mbuf[(*p).mesh];
      let ri = (*p).x - m.ci;
      (*p).v += min(1.0, u.dt * 20.0 * u.damp) * (m.vi + cross(m.wi, ri) - (*p).v);
      }*/  


    (*p).xupd = (*p).x;
    (*p).vupd = (*p).v;

    cell_delete((*p).hash, pid);
    (*p).hash = to_hash((*p).x);
    cell_insert((*p).hash, pid);

}


const plane_normals = array<v3,6>(v3(1,0,0),v3(0,1,0),v3(0,0,1),v3(-1,0,0),v3(0,-1,0),v3(0,0,-1));

@compute @workgroup_size(${threads})
fn collide(@builtin(global_invocation_id) gid:vec3<u32>) {
    let ipid = gid.x;
    if (ipid >= arrayLength(&pbuf)) { return; }
    let ip = &pbuf[ipid];
    let hash = (*ip).hash;
    let iw = (*ip).w;
    let iec = u.collision * (*ip).collision;
    let ief = u.friction * (*ip).friction;
    var ix = (*ip).x;
    var iv = (*ip).v;
    let ixp = (*ip).xprev;
    let ivp = (*ip).vprev;

    if (iw != 0) { 
        for (var z = -1; z <= 1; z++) {
            for (var y = -1; y <= 1; y++) {
                for (var x = -1; x <= 1; x++) {
                    let h = hash + dot(v3i(x,y,z), cell_stride);
                    let pids = &cells[h];
                    for (var j = 0; j < ${CELLSIZE}; j++) {
                        let jpid = atomicLoad(&((*pids)[j]));                    
                        if (jpid == -1 || u32(jpid) == ipid) { continue; }
                        let jp = &pbuf[jpid];
                        let jx = (*jp).x;
                        let x = ix - jx;
                        let l = length(x);
                        let dist = 2*u.r - l;
                        if (dist <= 0) { continue; }
                        let w = iw + (*jp).w;
                        if (w == 0) { continue; }                        
                        let n = select(x/l, v3(0,0,1), l == 0);
                        let jxp = (*jp).xprev;
                        let xp = ixp - jxp;
                        let dxn = min(0, 2*u.r - length(xp));
                        let jv = (*jp).v;
                        let jvp = (*jp).vprev;
                        let v = iv - jv;
                        let vp = ivp - jvp;
                        let vpn = dot(vp, n);
                        let a = (v - vp)/u.dt;
                        let an = dot(a, n);
                        var tc:f32;
                        if (an != 0) {
                            let discrim = vpn*vpn + 2*an*dxn;
                            if (discrim < 0) { continue; }
                            tc = (-vpn - sqrt(discrim))/an;
                        } else {
                            if (vpn == 0) { continue; }
                            tc = dxn / vpn;            
                        }
                        let tr = u.dt - tc;
                        let vc1 = vp + a * tc;
                        let vcn1 = dot(vc1, n);
                        let vct = vc1 - n*vcn1;
                        let ec = iec * (*jp).collision;
                        let ef = pow(ief * (*jp).friction, 3.0);
                        let vc2 = vc1 + n*(max(-vcn1*ec, -.5*an*tr) - vcn1) - ef*vct;
                        let xc = xp + (vp + vc1) * tc/2;
                        let dv = vc2 + a*tr;
                        let dx = xc + vc2*tr + .5*a*tr*tr;
                        ix += iw/w * dx;
                        iv += iw/w * dv;
                    }                
                }
            }
        }
    }
    
    for (var i = 0; i < 6; i++) {
        let n = plane_normals[i];
        let point = select(u.spacemin, u.spacemax, i >= 3);
        let dist = u.r - dot(ix - point, n);
        if (dist <= 0) { continue; }
        var dxn = min(0,u.r - dot(ixp - point, n));        
        let vpn = dot(ivp, n);
        let a = (iv - ivp)/u.dt;
        let an = dot(a, n);
        var tc:f32;
        if (an != 0) {
            let discrim = vpn*vpn + 2*an*dxn;
            if (discrim < 0) { continue; }
            tc = (-vpn - sqrt(discrim))/an;
        } else {
            if (vpn == 0) { continue; }
            tc = dxn / vpn;            
        }        
        let tr = u.dt - tc;
        let vc1 = ivp + a * tc;
        let vcn1 = dot(vc1, n);
        let vct = vc1 - n*vcn1;
        let vc2 = vc1 + n*(max(-vcn1*iec, -.5*an*tr) - vcn1) - pow(ief, 3.0)*vct;
        let xc = ixp + (ivp + vc1) * tc/2;
        iv = vc2 + a*tr;
        ix = xc + vc2*tr + .5*a*tr*tr;
    }

    (*ip).xupd = ix;
    (*ip).vupd = iv;
}

@compute @workgroup_size(${threads})
fn xvupd(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid = gid.x;
    if (pid >= arrayLength(&pbuf)) { return; }
    let p = &pbuf[pid];
    if ((*p).w != 0) {
        (*p).x = (*p).xupd;
        (*p).v = (*p).vupd;
        (*p).xprev = (*p).xupd - (*p).v * u.dt;
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
@compute @workgroup_size(1)
fn surfmatch(@builtin(global_invocation_id) gid:vec3<u32>) {
    
    let nclusters = arrayLength(&cbuf);
    for (var cid = 0u; cid < nclusters; cid++) {
        
        let cluster = cbuf[cid];
        let m = mbuf[cluster.mesh];
        var volstiff = u.volstiff * m.volstiff;
        var shearstiff = u.shearstiff * m.shearstiff;
        if (volstiff == 0 || shearstiff == 0) { continue; }
        shearstiff = 1.0/shearstiff - 1.0;
        volstiff = 0.01 * (1.0/volstiff - 1.0);
           
        var n = cluster.n;
        if (n < 4) { continue; }
        var pids = cluster.pids;
        let s = cluster.s;
        let Qinv = cluster.qinv;
        
        let c0 = cluster.c0;
        var c = v3();
        for (var i = 0u; i < n; i++) {
            c += pbuf[pids[i]].x;
        }
        c /= f32(n);
        
        var P = m3();
        for (var i = 0u; i < n; i++) {
            let p = &pbuf[pids[i]];
            let rs = s*((*p).x - c);
            let r0 = (*p).x0 - c0;
            P += m3(rs*r0.x, rs*r0.y, rs*r0.z);
        }
        
        var F = P * Qinv;
        var C = sqrt(dot(F[0],F[0]) + dot(F[1],F[1]) + dot(F[2],F[2]));
        if (C == 0) { continue; }
        
        var G = s/C * (F * transpose(Qinv));
        var walpha = shearstiff / u.dt / u.dt;
        for (var i = 0u; i < n; i++) {
            let grad = G * (pbuf[pids[i]].x0 - c0);
            walpha += dot(grad,grad);
        }
        
        c = v3();
        var lambda = select(-C / walpha, 0.0, walpha == 0.0);
        for (var i = 0u; i < n; i++) {
            let p = &pbuf[pids[i]];
            let x = (*p).x + lambda * (G * ((*p).x0 - c0));
            c += x;
            if ((*p).w != 0) { (*p).x = x; }
        }
        c /= f32(n);
        
        P = m3();
        for (var i = 0u; i < n; i++) {
            let p = &pbuf[pids[i]];
            let rs = s*((*p).x - c);
            let r0 = (*p).x0 - c0;
            P += m3(rs*r0.x, rs*r0.y, rs*r0.z);
        }
        F = P * Qinv;
        C = determinant(F) - 1.0;
        
        G = s * m3(cross(F[1],F[2]),cross(F[2],F[0]),cross(F[0],F[1])) * transpose(Qinv);
        walpha = volstiff / u.dt / u.dt;
        for (var i = 0u; i < n; i++) {
            let grad = G * (pbuf[pids[i]].x0 - c0);
            walpha += dot(grad,grad);
        }
        
        c = v3();
        lambda = select(-C / walpha, 0.0, walpha == 0.0);
        for (var i = 0u; i < n; i++) {
            let p = &pbuf[pids[i]];
            let x = (*p).x + lambda * (G * ((*p).x0 - c0));
            c += x;
            if ((*p).w != 0) { (*p).x = x; }
        }
        c /= f32(n);
        
        P = m3();
        for (var i = 0u; i < n; i++) {
            let p = &pbuf[pids[i]];
            let rs = s*((*p).x - c);
            let r0 = (*p).x0 - c0;
            P += m3(rs*r0.x, rs*r0.y, rs*r0.z);
        }
        
        F = P * Qinv;
        for (var i = 0u; i < n; i++) {
            let p = &pbuf[pids[i]];
            if ((*p).w != 0) {
                (*p).x = c + F * ((*p).x0 - c0);
            } 
        }
    }
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




    
