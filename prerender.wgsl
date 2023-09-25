alias v2 = vec2<f32>;
alias v2i = vec2<i32>;
alias v3 = vec3<f32>;
alias v4 = vec4<f32>;

@compute @workgroup_size(${threads})
fn update_tris(@builtin(global_invocation_id) gid:vec3<u32>) {
    let tid:u32 = gid.x;
    let ntris = arrayLength(&tribuf);
    if (tid >= ntris) { return; }
    var t = tribuf[tid];
    t.v0.x = pbuf[t.v0.pidx].x;
    t.v1.x = pbuf[t.v1.pidx].x;
    t.v2.x = pbuf[t.v2.pidx].x;
    t.v0.norm = vbuf[t.v0.pidx].norm;
    t.v1.norm = vbuf[t.v1.pidx].norm;
    t.v2.norm = vbuf[t.v2.pidx].norm;
    tribuf[tid] = t;
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
    let v = &vbuf[pid];
    var norm = v3(0f,0f,0f);
    let nedges = (*v).nedges;
    let x = (*p).x;
    for (var i = 0u; i < nedges; i = i + 1u) {
        var ab = pbuf[(*v).edges[i % nedges]].x - x;
        var ac = pbuf[(*v).edges[(i+1u) % nedges]].x - x;
        var weight = 1.0;
        //weight *= length(cross(ab,ac))/2; // area
        weight *= acos(dot(safenorm(ab),safenorm(ac))); // angle
        let n = safenorm(cross(ab,ac));
        norm += weight * n;
    }
    (*v).norm = safenorm(norm);
}






