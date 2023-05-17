alias v2 = vec2<f32>;
alias v2i = vec2<i32>;
alias v3 = vec3<f32>;
alias v4 = vec4<f32>;


@compute @workgroup_size(${threads})
fn vertpos(@builtin(global_invocation_id) gid:vec3<u32>) {
    let vid:u32 = gid.x;
    let nverts = arrayLength(&vertices);
    if (vid >= nverts) { return; }
    var v = vertices[vid];
    let tid = v.tet;
    if (tid >= 0) {
        let tet = tets[tid];
        var u = 1.0;
        v.pos = v3(0);
        for (var k = 0; k < 3; k += 1) {
            v.pos += v.bary[k] * particles[tet[k + 1]].pos;
            u -= v.bary[k];
        }
        v.pos += u * particles[tet[0]].pos;
    }
    vertices[vid] = v;
}

@compute @workgroup_size(${threads})
fn update_tris(@builtin(global_invocation_id) gid:vec3<u32>) {
    let tid:u32 = gid.x;
    let ntris = arrayLength(&tris);
    if (tid >= ntris) { return; }
    var tri = tris[tid];
    let v0 = vertices[tri.v0.vidx];
    let v1 = vertices[tri.v1.vidx];
    let v2 = vertices[tri.v2.vidx];
    tri.v0.pos = v0.pos;
    tri.v1.pos = v1.pos;
    tri.v2.pos = v2.pos;
    tri.v0.norm = v0.norm;
    tri.v1.norm = v1.norm;
    tri.v2.norm = v2.norm;
    tri.v0.mesh = v0.mesh;
    tri.v1.mesh = v1.mesh;
    tri.v2.mesh = v2.mesh;
    tris[tid] = tri;
}

@compute @workgroup_size(${threads})
fn normals(@builtin(global_invocation_id) gid:vec3<u32>) {
    let vid:u32 = gid.x;
    let nverts = arrayLength(&vertices);
    if (vid >= nverts) { return; }
    let v = vertices[vid];
    let nedges = v.nedges;
    let pos = v.pos;
    var norm = v3(0f,0f,0f);
    for (var i = 0u; i < nedges; i = i + 1u) {
        let ab = vertices[v.edges[i % nedges]].pos - pos;
        let ac = vertices[v.edges[(i+1u) % nedges]].pos - pos;
        let n = normalize(cross(ab,ac));
        let angle = acos(dot(normalize(ab),normalize(ac)));
        norm += angle * n;
    }
    if (length(norm) > 0) {
    	vertices[vid].norm = normalize(norm);
    }
}

fn clipToPixel(clip:v2, dim:v2i) -> v2i {
    return v2i(i32(f32(dim.x) * (1+clip.x)/2),
               i32(f32(dim.y) * (1-clip.y)/2));
}




