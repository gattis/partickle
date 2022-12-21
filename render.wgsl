type v2 = vec2<f32>;
type v2i = vec2<i32>;
type v4 = vec4<f32>;
type v3 = vec3<f32>;
type m3 = mat3x3<f32>;
type m4 = mat4x4<f32>;

const shininess = 16.0;
const ambient = 0.2f;
const cube = array<v3,14>(v3(-1,1,1), v3(1,1,1), v3(-1,-1,1), v3(1,-1,1), v3(1,-1,-1),
                          v3(1,1,1), v3(1,1,-1), v3(-1,1,1), v3(-1,1,-1), v3(-1,-1,1),
                          v3(-1,-1,-1), v3(1,-1,-1), v3(-1,1,-1), v3(1,1,-1));



fn frag(worldpos:v3, norm:v3, color:v4) -> v4 {
    var mix = color.rgb * ambient;
    for (var i = 0; i < ${numLights}; i += 1) {
        let light = &lights[i];
        var lightdir = (*light).pos - worldpos;
        let distance = length(lightdir);
        let lightmag = (*light).color * (*light).power / (1.0 + distance);
        lightdir = normalize(lightdir);


        let lambertian = max(dot(lightdir, norm), 0.0);
        var specular = 0.0f;
        if (lambertian > 0.0) {
            let viewdir = normalize(uniforms.cam_pos - worldpos);
            let reflectdir = reflect(-lightdir, norm);
            let specAngle = max(dot(reflectdir, viewdir), 0.0);
            specular = pow(specAngle, shininess);
        }
        mix += lightmag * (color.rgb*lambertian + specular);
    }

    return v4(clamp(mix,v3(0),v3(1)), color.a);

}

struct SurfOut {
    @builtin(position) position: v4,
    @location(0) worldpos:v3,
    @location(1) norm:v3,
    @location(2) uv:v2,
    @location(3) @interpolate(flat) mesh:u32,
};

@vertex fn surface_vert(@location(0) pos:v3,
                     @location(1) norm:v3,
                     @location(2) mesh:u32,
                     @location(3) uv:v2) -> SurfOut {
    var out:SurfOut;
    out.worldpos = pos;
    out.position = uniforms.mvp * v4(out.worldpos, 1.0);
    out.norm = norm;
    out.uv = uv;
    out.mesh = mesh;
    return out;
}

struct SurfIn {
    @builtin(position) position: v4,
    @builtin(front_facing) front:bool,
    @location(0) worldpos:v3,
    @location(1) norm:v3,
    @location(2) uv:v2,
    @location(3) @interpolate(flat) mesh:u32,
}

struct FragDepth {
    @location(0) color: v4,
    @builtin(frag_depth) depth:f32
};


@fragment fn surface_frag(input:SurfIn) -> @location(0) v4 {
    let m = &meshes[input.mesh];
    if (bool((*m).inactive)) { discard; }
    if ((*m).fluid == 1) { discard; }
    var color = (*m).color * select(textureSample(tex, samp, input.uv, (*m).tex), v4(1), (*m).tex < 0);
    if (color.a < 0.0001) { discard; }
    color = frag(input.worldpos, input.norm, color);
    return v4(color.rgb, color.a);
}


@vertex fn axes_vert(@builtin(vertex_index) vertidx:u32,
                     @builtin(instance_index) instidx:u32) -> @builtin(position) v4 {
    var worldPos = v3(0);
    worldPos[instidx] = f32(2*i32(vertidx) - 1);
    return uniforms.mvp * v4(worldPos, 1.0);
}

@fragment fn axes_frags() -> @location(0) v4 {
    return v4(1,1,1,1);
}

fn normals_vert(vertidx:u32, vertPos:v3, norm:v3) -> v4 {
    var worldPos = vertPos;
    if (vertidx == 1u) {
        worldPos += norm * 0.05;
    }
    return uniforms.mvp * v4(worldPos, 1.0);
}


@vertex fn vnormals_vert(@builtin(vertex_index) vertidx:u32, @location(0) vertPos:v3,
                         @location(1) norm:v3) -> @builtin(position) v4 {
    return normals_vert(vertidx, vertPos, norm);
}

@vertex fn pnormals_vert(@builtin(vertex_index) vertidx:u32, @location(0) partPos:v3,
                         @location(1) norm:v3) -> @builtin(position) v4 {
    return normals_vert(vertidx, partPos, norm);
}

@fragment fn pnormals_frag() -> @location(0) v4 {
    return v4(1,1,1,1);
}

@fragment fn vnormals_frag() -> @location(0) v4 {
    return v4(.2,1,.1,0);
}

@vertex fn edges_vert(@builtin(vertex_index) vertidx:u32,
                      @location(0) pos:v3) -> @builtin(position) v4 {
    let a = f32(vertidx);
    return uniforms.mvp * v4(pos, 1.0);
}

@fragment fn edges_frag() -> @location(0) v4 {
    return v4(1,1,1,1);
}


struct Impostor {
    vertpos:v3,
    uv:v2
};

const sq3 = 1.73205077648;

fn impostor(vertidx:u32, pos:v3, r:f32) -> Impostor {
    let fwd = normalize(pos - uniforms.cam_pos);
    let right = normalize(v3(-fwd.y, fwd.x, 0));
    let up = normalize(cross(fwd,right));
    if (vertidx == 0) {
        return Impostor(r * 2 * up, v2(0, 2));
    } else if (vertidx == 1) {
        return Impostor(r * (sq3*right - up), v2(sq3, -1));
    }
    return Impostor(r * (-sq3*right - up), v2(-sq3, -1));
}


struct PartIO {
    @builtin(position) position:v4,
    @location(0) partpos:v3,
    @location(1) vertpos:v3,
    @location(2) uv:v2,
    @location(3) @interpolate(flat) mesh:u32,
    @location(4) @interpolate(flat) selected:u32,
};

@vertex fn particle_vert(@builtin(vertex_index) vertidx:u32,
                     @builtin(instance_index) instidx:u32,
                     @location(0) partpos:v3,
                     @location(1) mesh:u32) -> PartIO {

    var out:PartIO;
    let imp = impostor(vertidx, partpos, uniforms.r);
    out.partpos = partpos;
    out.vertpos = imp.vertpos;
    out.position = uniforms.mvp * v4(out.partpos + out.vertpos,1);
    out.mesh = mesh;
    out.selected = select(0u, 1u, i32(instidx) == uniforms.selection);
    return out;
}


struct FluidIO {
    @builtin(position) position:v4,
    @location(0) partpos:v3,
    @location(1) vertpos:v3,
    @location(2) uv:v2,
    @location(3) @interpolate(flat) instidx:u32,
};

@vertex fn fluid_vert(@builtin(vertex_index) vertidx:u32,
                      @builtin(instance_index) instidx:u32,
                      @location(0) partpos:v3) -> FluidIO {
    var out:FluidIO;
    let imp = impostor(vertidx, partpos, uniforms.r*2);
    out.vertpos = imp.vertpos;
    out.partpos = partpos;
    out.uv = imp.uv;
    out.position = uniforms.mvp * v4(partpos + out.vertpos,1);
    out.instidx = instidx;
    return out;
}

@fragment fn fluid_frag(input:FluidIO) -> @location(0) v4 {
    let p = particles[input.instidx];
    let m = meshes[p.mesh];
    if (bool(m.inactive)) { discard; }
    if (m.fluid != 1) { discard; }
    let r = length(input.uv);
    if (r > 1.0) { discard; }
    var mag = clamp(pow(1.0/(.4+r),8),0,1);
    if (p.k < 2) { discard; }

    let color = m.color.rgb * sin(uniforms.t/10);
    return v4(color, m.color.a*mag);
}


struct LightIO {
    @builtin(position) position:v4,
    @location(0) lightpos:v3,
    @location(1) vertpos:v3,
    @location(2) uv:v2,
    @location(3) color:v3,
    @location(4) size:f32,
};

@vertex fn lights_vert(@builtin(vertex_index) vertidx:u32,
                      @builtin(instance_index) instidx:u32) -> LightIO {
    let l = &lights[instidx];
    var out:LightIO;
    out.lightpos = (*l).pos;
    out.size = .4*sqrt((*l).power);
    let imp = impostor(vertidx, out.lightpos, out.size);
    out.vertpos = imp.vertpos;
    out.position = uniforms.mvp * v4(out.lightpos + out.vertpos,1);
    out.color = (*l).color;
    return out;
}


struct RayTrace {
    t:f32,
    rd:v3,
    hit:v3,
    normal:v3,
    clip_depth: f32
};


fn trace_sphere(vertpos:v3, center:v3, r:f32) -> RayTrace {
    var trace:RayTrace;
    trace.rd = normalize(vertpos + center - uniforms.cam_pos);
    let co = uniforms.cam_pos - center;
    let b = dot(co, trace.rd);
    let c = b*b - dot(co, co) + r*r;
    if (c < 0) { discard; }
    let t1 = -b + sqrt(c);
    let t2 = -b - sqrt(c);
    if (t1 >= 0 && t2 >= 0) { trace.t = min(t1,t2); }
    else if (t1 >= 0 && t2 < 0) { trace.t = t1; }
    else if (t1 < 0 && t2 >= 0) { trace.t = t2; }
    else { discard; }
    let ot = trace.rd * trace.t;
    trace.hit = uniforms.cam_pos + ot;
    trace.normal = normalize(ot + co);
    let hitclip = uniforms.mvp * v4(trace.hit, 1);
    trace.clip_depth = hitclip.z / hitclip.w;
    return trace;
}

@fragment fn particle_frag(input:PartIO) -> FragDepth {
    let m = &meshes[input.mesh];
    if (bool((*m).inactive)) { discard; }
    let color = (*m).pcolor;
    if (color.a < 0.5) { discard; }
    var rgb = color.rgb;
    if (input.selected == 1u) {
        rgb = 1 - rgb;
    }
    let trace = trace_sphere(input.vertpos, input.partpos, uniforms.r);
    return FragDepth(frag(trace.hit, trace.normal, v4(rgb,1.0f)), trace.clip_depth);
}


@fragment fn lights_frag(input:LightIO) -> FragDepth {
    let trace = trace_sphere(input.vertpos, input.lightpos, input.size);
    var mag = .01/(1-dot(-trace.normal, trace.rd));
    return FragDepth(v4(input.color*mag,mag), trace.clip_depth);
}


struct GndIO {
    @builtin(position) position:v4,
    @location(0) vertpos:v3,
};

const rgnd = 1000.0f;
const gnd_color = v4(1, 1, 1, 1);


@vertex fn ground_vert(@builtin(vertex_index) vertidx:u32) -> GndIO {
    var out:GndIO;
    let vpos = cube[vertidx];
    out.vertpos = vpos*rgnd;
    out.position = uniforms.mvp * v4(rgnd * (vpos - v3(0,0,1)), 1);
    return out;
}


fn checkers(xy:v2) -> f32 {
    return f32(abs(i32(floor(xy.x)) + i32(floor(xy.y))) % 2);
}

@fragment fn ground_frag(input:GndIO) -> FragDepth {
    let trace = trace_sphere(input.vertpos, v3(0,0,-rgnd), rgnd);
    let pattern = .2*checkers(trace.hit.xy/.1) + .3*checkers(trace.hit.xy) + 0.1;
    let fade = 10/(10 + length(trace.hit.xy));
    var color = v4(v3(clamp(pattern,.2,.8)*fade),1);

    return FragDepth(frag(trace.hit, trace.normal, color), trace.clip_depth);
}
