type v3 = vec3<f32>;

@compute @workgroup_size(${threads})
fn vertpos(@builtin(global_invocation_id) gid:vec3<u32>) {
    let vid:u32 = gid.x;
    let nverts = arrayLength(&vertices);
    if (vid >= nverts) { return; }
    let v = &vertices[vid];
    let p = &particles[(*v).particle];
    let m = &meshes[(*v).mesh];
    let goal_delta = (*p).si - ((*m).ci + (*m).rot * (*p).q);
    (*v).pos = (*m).ci + (*m).rot * (*v).q +  goal_delta;
}

fn closest_tri_delta(a:v3, b:v3, c:v3, p:v3) -> v3 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let bp = p - b;
    let d1 = dot(ab,ap);
    let d2 = dot(ac,ap);
    if (d1 <= 0 && d2 <= 0) { return ap; }
    let d3 = dot(ab,bp);
    let d4 = dot(ac,bp);
    if (d3 >= 0 && d4 <= d3) { return bp; }
    let vc = d1*d4 - d3*d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) { return ap - ab*(d1/(d1-d3)); }
    let cp = p - c;
    let d5 = dot(ab,cp);
    let d6 = dot(ac,cp);
    if (d6 >= 0 && d5 <= d6) { return cp; }
    let vb = d5*d2 - d1*d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) { return ap - ac*(d2/(d2-d6)); }
    let va = d3*d6 - d5*d4;
    let bc = c - b;
    let d7 = d4 - d3;
    let d8 = d5 - d6;
    if (va <= 0 && d7 >= 0 && d8 >= 0) { return bp - bc*(d7/(d7+d8)); }       
    return ap - (ab*vb + ac*vc)/(va + vb + vc);
}

@compute @workgroup_size(${threads})
fn update_tris(@builtin(global_invocation_id) gid:vec3<u32>) {
    let tid:u32 = gid.x;
    let ntris = arrayLength(&tris);
    if (tid >= ntris) { return; }
    let tri = &tris[tid];
    if ((*tri).v0.vidx == 0 && (*tri).v1.vidx == 0 && (*tri).v2.vidx == 0) {
        (*tri).v0.dist = -1e38;
        (*tri).v1.dist = -1e38;
        (*tri).v2.dist = -1e38;
        return;
    }
    let v0 = &vertices[(*tri).v0.vidx];
    let v1 = &vertices[(*tri).v1.vidx];
    let v2 = &vertices[(*tri).v2.vidx];
    (*tri).v0.pos = (*v0).pos;
    (*tri).v1.pos = (*v1).pos;
    (*tri).v2.pos = (*v2).pos;
    (*tri).v0.norm = (*v0).norm;
    (*tri).v1.norm = (*v1).norm;
    (*tri).v2.norm = (*v2).norm;
    (*tri).v0.mesh = (*v0).mesh;
    (*tri).v1.mesh = (*v1).mesh;
    (*tri).v2.mesh = (*v2).mesh;

}

/*@compute @workgroup_size(${threads})
fn sort_tris(@builtin(global_invocation_id) gid:vec3<u32>) {
    let n = arrayLength(&tris);
    let i = gid.x;
    let j = i ^ tonic.y;
    if (i >= j || i >= n || j >= n) { return; }
    let itri = tris[i];
    let jtri = tris[j];

    if (((i & tonic.x) == 0u) == (itri.v0.dist < jtri.v0.dist)) {
        tris[i] = jtri;
        tris[j] = itri;
    }
    }*/


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

}

