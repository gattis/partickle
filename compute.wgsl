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

fn safenorm(v:v3) -> v3 {
    let l = length(v);
    return select(v/l, v3(0,0,0), l == 0);
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

fn x2cell(x:v3) -> v3i {
    return v3i(floor(x / (2*u.r)));
}

fn cell2hash(cell:v3i) -> i32 {
    return dot((cell % ${CELLDIM} + ${CELLDIM}) % ${CELLDIM}, cell_stride);
}

fn x2hash(x:v3) -> i32 {
    return cell2hash(x2cell(x));
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

    if (u.grab_pid == i32(pid)) { // grabbing
        (*p).v = .7*(*p).v + .3*(u.grab_x - (*p).x);
        (*p).x = u.grab_x;
        (*p).w = 0;
    } else if ((*p).w == 0 && w0[pid] != 0) { // dropped
        (*p).v = .7*(*p).v + .3*(u.grab_x - (*p).x);
        //(*p).v = u.grab_x - (*p).x;
        (*p).x = u.grab_x;
        (*p).w = w0[pid];
    } else if (w0[pid] == 0) { // fixed
        (*p).v = v3(0);
        (*p).w = 0;
    } else { // usual
        let vprev = (*p).v;        
        (*p).v += u.a;
        (*p).x += ((*p).v + vprev)/2;
        (*p).w = w0[pid];
    }

    /*
      if ((*p).nedges != 0) {
      let m = mbuf[(*p).mesh];
      let ri = (*p).x - m.ci;
      (*p).v += min(1.0, 20.0 * u.damp) * (m.vi + cross(m.wi, ri) - (*p).v);
      }*/  


    (*p).dx = v3(0);
    cell_delete((*p).hash, pid);
    (*p).hash = x2hash((*p).x);
    cell_insert((*p).hash, pid);

}


const plane_normals = array<v3,6>(v3(1,0,0),v3(0,1,0),v3(0,0,1),v3(-1,0,0),v3(0,-1,0),v3(0,0,-1));

@compute @workgroup_size(${threads})
fn wall_collide(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid = gid.x;
    if (pid >= arrayLength(&pbuf)) { return; }
    let p = &pbuf[pid];
    let ec = u.collision * (*p).collision;
    let ef = pow(u.friction * (*p).friction, 3.0);
    let x = (*p).x;
    let v = (*p).v;
    var dx = v3(0,0,0);
    var dv = v3(0,0,0);
    
    for (var plane = 0; plane < 6; plane++) {
        let n = plane_normals[plane];
        let point = select(u.spacemin, u.spacemax, plane >= 3) + u.r*n;
        var xn = dot(x-point, n);
        if (xn >= 0) { continue; }
        dx -= xn*n;
        let vn1 = dot(v,n);
        let an = dot(u.a,n);
        let disc = vn1*vn1 - 2*an*xn;
        let vn2 = select(ec * sqrt(disc), 0, disc <= 0);
        let vt = v - n*vn1;
        let vtmag = length(vt);
        let vtdir = select(vt/vtmag, v3(0,0,0), vtmag == 0);
        let an2 = vn2 - vn1;
        dv += an2*n - min(ef*an2,vtmag)*vtdir;
    }
   
    (*p).x += dx;
    (*p).v += dv;

}



@compute @workgroup_size(${threads})
fn pair_collide(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid = gid.x;
    if (pid >= arrayLength(&pbuf)) { return; }
    let p = &pbuf[pid];
    let pw = (*p).w;
    if (pw == 0) { return; }
    let D = 2*u.r;
    let pec = u.collision * (*p).collision;
    let pef = u.friction * (*p).friction;
    let px = (*p).x;
    let pv = (*p).v;
    let cell = x2cell(px);
    var dx = v3(0,0,0);
    var dv = v3(0,0,0);
    for (var cz = -1; cz <= 1; cz++) {
        for (var cy = -1; cy <= 1; cy++) {
            for (var cx = -1; cx <= 1; cx++) {
                let h = cell2hash(cell + v3i(cx,cy,cz));
                let pids = &cells[h];
                for (var i = 0; i < ${CELLSIZE}; i++) {
                    let qid = atomicLoad(&((*pids)[i]));
                    if (qid == -1 || u32(qid) == pid) { continue; }
                    let q = &pbuf[qid];
                    let w = pw + (*q).w;
                    if (w == 0) { continue; }
                    let x = px - (*q).x;
                    let l = length(x);
                    if (l >= D) { continue; }
                    let v = pv - (*q).v;
                    var n = select(x/l, select(v3(1,0,0),v3(-1,0,0),pid < u32(qid)), l == 0);
                    let ec = pec * (*q).collision;
                    let vn = dot(v,n);
                    let pvn = dot(pv,n);
                    let pvn2 = ec * (pvn - 2*pw/w*vn);
                    let dvn = pvn2 - pvn;
                    dv += dvn*n;
                    dx += pw/w * (2*u.r - l) * n;
                }                
            }
        }
    }  
    
    (*p).dx = dx;
    (*p).dv = dv;

}

@compute @workgroup_size(${threads})
fn xvupd(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid = gid.x;
    if (pid >= arrayLength(&pbuf)) { return; }
    let p = &pbuf[pid];
    if ((*p).w != 0) {
        (*p).x += (*p).dx;
        (*p).v += (*p).dv;        
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
        var walpha = shearstiff;
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
        walpha = volstiff;
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
        let v = pbuf[i].v;
        vavg[gid.x] = v;
        let r = pbuf[i].x - m.ci;
        let R = m3(0, r.z, -r.y, -r.z, 0, r.x, r.y, -r.x, 0);
        lavg[gid.x] = cross(r, v);
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




    
