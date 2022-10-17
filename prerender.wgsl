type v3 = vec3<f32>;

@compute @workgroup_size(${threads})
fn vertpos(@builtin(global_invocation_id) gid:vec3<u32>) {
    let vid:u32 = gid.x;
    let nverts = arrayLength(&vertices);
    if (vid >= nverts) { return; }
    let v = &vertices[vid];
    let m = &meshes[(*v).mesh];
    let pid = (*v).particle;
    let tid = (*v).tet;
    if (tid >= 0) {
        let bary = (*v).bary;
        let tet = tets[tid];
        var u = 1.0;
        (*v).pos = v3(0);
        for (var k = 0; k < 3; k += 1) {
            (*v).pos += bary[k] * particles[tet.verts[k]].pos;
            u -= bary[k];
        }
        (*v).pos += u * particles[tet.verts[3]].pos;
    } else if (pid > 0) {
        let p = &particles[pid];       
        let goal_delta = (*p).pos - ((*m).ci + (*m).rot * (*p).q);
        (*v).pos = (*m).ci + (*m).rot * (*v).q +  goal_delta;
    } // else {  (*v).pos = (*m).ci + (*m).rot * (*v).q; }

}

@compute @workgroup_size(${threads})
fn update_tris(@builtin(global_invocation_id) gid:vec3<u32>) {
    let tid:u32 = gid.x;
    let ntris = arrayLength(&tris);
    if (tid >= ntris) { return; }
    let tri = &tris[tid];
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

