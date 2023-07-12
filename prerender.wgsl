alias v2 = vec2<f32>;
alias v2i = vec2<i32>;
alias v3 = vec3<f32>;
alias v4 = vec4<f32>;

@compute @workgroup_size(${threads})
fn update_tris(@builtin(global_invocation_id) gid:vec3<u32>) {
    let tid:u32 = gid.x;
    let ntris = arrayLength(&tribuf);
    if (tid >= ntris) { return; }
    var tri = tribuf[tid];
    let p0 = &pbuf[tri.v0.pidx];
    let p1 = &pbuf[tri.v1.pidx];
    let p2 = &pbuf[tri.v2.pidx];
    tri.v0.x = (*p0).x;
    tri.v1.x = (*p1).x;
    tri.v2.x = (*p2).x;
    tri.v0.norm = (*p0).norm;
    tri.v1.norm = (*p1).norm;
    tri.v2.norm = (*p2).norm;
    tri.v0.mesh = (*p0).mesh;
    tri.v1.mesh = (*p1).mesh;
    tri.v2.mesh = (*p2).mesh;
    tribuf[tid] = tri;
}

fn safenorm(v:v3) -> v3 {
    let l = length(v);
    if (l == 0.) { return v; }
    return v / l;
}

@compute @workgroup_size(${threads})
fn normals(@builtin(global_invocation_id) gid:vec3<u32>) {
    let pid:u32 = gid.x;
    let nparts = arrayLength(&pbuf);
    if (pid >= nparts) { return; }
    let p = &pbuf[pid];
    var norm = v3(0f,0f,0f);
    let nedges = (*p).nedges;
    let x = (*p).x;
    for (var i = 0u; i < nedges; i = i + 1u) {
        var ab = pbuf[(*p).edges[i % nedges]].x - x;
        var ac = pbuf[(*p).edges[(i+1u) % nedges]].x - x;
        var weight = 1.0;
        //weight *= length(cross(ab,ac))/2; // area
        weight *= acos(dot(safenorm(ab),safenorm(ac))); // angle
        let n = safenorm(cross(ab,ac));
        norm += weight * n;
    }
    (*p).norm = safenorm(norm);
}






