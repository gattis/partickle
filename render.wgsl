alias v2 = vec2<f32>;
alias v2i = vec2<i32>;
alias v4 = vec4<f32>;
alias v3 = vec3<f32>;
alias m3 = mat3x3<f32>;
alias m4 = mat4x4<f32>;

const shininess = 16.0;
const ambient = 0.2f;
const cube = array<v3,14>(v3(-1,1,1), v3(1,1,1), v3(-1,-1,1), v3(1,-1,1), v3(1,-1,-1),
                          v3(1,1,1), v3(1,1,-1), v3(-1,1,1), v3(-1,1,-1), v3(-1,-1,1),
                          v3(-1,-1,-1), v3(1,-1,-1), v3(-1,1,-1), v3(1,1,-1));

fn frag(worldx:v3, norm:v3, color:v4) -> v4 {
    var mix = color.rgb * ambient;
    for (var i = 0; i < ${numLights}; i += 1) {
        let light = lbuf[i];
        var lightdir = light.x - worldx;
        let distance = length(lightdir);
        let lightmag = light.color * light.power / (1.0 + distance);
        lightdir = normalize(lightdir);

        let lambertian = max(dot(lightdir, norm), 0.0);
        var specular = 0.0f;
        if (lambertian > 0.0) {
            let viewdir = normalize(uni.cam_x - worldx);
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
    @location(0) worldx:v3,
    @location(1) norm:v3,
    @location(2) uv:v2,
    @location(3) @interpolate(flat) mesh:u32,
};

@vertex fn surface_vert(@location(0) x:v3,
                        @location(1) norm:v3,
                        @location(2) mesh:u32,
                        @location(3) uv:v2) -> SurfOut {
    var out:SurfOut;
    out.worldx = x;
    out.position = uni.mvp * v4(out.worldx, 1.0);
    out.norm = norm;
    out.uv = uv;
    out.mesh = mesh;
    return out;
}

struct SurfIn {
    @builtin(position) position: v4,
    @builtin(front_facing) front:bool,
    @location(0) worldx:v3,
    @location(1) norm:v3,
    @location(2) uv:v2,
    @location(3) @interpolate(flat) mesh:u32,
}

struct FragDepth {
    @location(0) color: v4,
    @builtin(frag_depth) depth:f32
};

@fragment fn surface_frag(input:SurfIn) -> @location(0) v4 {
    let m = mbuf[input.mesh];
    if (m.fluid == 1) { discard; }
    var color = m.color * select(textureSample(tex, samp, input.uv, m.tex), v4(1), m.tex < 0);
    if (color.a < 0.0001) { discard; }
    color = frag(input.worldx, input.norm, color);
    return v4(color.rgb, color.a);
}

@vertex fn vnormals_vert(@builtin(vertex_index) vertidx:u32, @location(0) vertx:v3,
                         @location(1) norm:v3) -> @builtin(position) v4 {
    var worldx = vertx;
    if (vertidx == 1u) {
        worldx += norm * 0.05;
    }
    return uni.mvp * v4(worldx, 1.0);
}

@fragment fn vnormals_frag() -> @location(0) v4 {
    return v4(.2,1,.1,1);
}

const sq3 = 1.73205077648;

fn impostor(vertidx:u32, x:v3, r:f32) -> v3 {
    let fwd = normalize(x - uni.cam_x);
    let right = normalize(v3(-fwd.y, fwd.x, 0));
    let up = normalize(cross(fwd,right));
    let z = -fwd;
    if (vertidx == 0) {
        return r*(2*up + z);
    } else if (vertidx == 1) {
        return r*(sq3*right - up + z);
    }
    return r*(-sq3*right - up + z);
}

struct PartIO {
    @builtin(position) position:v4,
    @location(0) partx:v3,
    @location(1) vertx:v3,
    @location(2) uv:v2,
    @location(3) @interpolate(flat) mesh:u32,
    @location(4) @interpolate(flat) selected:u32,
};

@vertex fn particle_vert(@builtin(vertex_index) vertidx:u32,
                         @builtin(instance_index) instidx:u32,
                         @location(0) partx:v3,
                         @location(1) mesh:u32) -> PartIO {
    var out:PartIO;
    let imp = impostor(vertidx, partx, uni.r);
    out.partx = partx;
    out.vertx = imp;
    out.position = uni.mvp * v4(out.partx + out.vertx,1);
    out.mesh = mesh;
    out.selected = select(0u, 1u, i32(instidx) == uni.selection);
    return out;
}

struct LightIO {
    @builtin(position) position:v4,
    @location(0) lightx:v3,
    @location(1) vertx:v3,
    @location(2) uv:v2,
    @location(3) color:v3,
    @location(4) size:f32,
};

@vertex fn lights_vert(@builtin(vertex_index) vertidx:u32,
                       @builtin(instance_index) instidx:u32) -> LightIO {
    let l = lbuf[instidx];
    var out:LightIO;
    out.lightx = l.x;
    out.size = .4*sqrt(l.power);
    let imp = impostor(vertidx, out.lightx, out.size);
    out.vertx = imp;
    out.position = uni.mvp * v4(out.lightx + out.vertx,1);
    out.color = l.color;
    return out;
}

struct RayTrace {
    t:f32,
    rd:v3,
    hit:v3,
    normal:v3,
    clip_depth: f32
};

fn trace_sphere(vertx:v3, center:v3, r:f32) -> RayTrace {
    var trace:RayTrace;
    trace.t = -1;
    trace.rd = normalize(vertx + center - uni.cam_x);
    let co = uni.cam_x - center;
    let b = dot(co, trace.rd);
    let c = b*b - dot(co, co) + r*r;
    if (c < 0) { return trace; }
    let t1 = -b + sqrt(c);
    let t2 = -b - sqrt(c);
    if (t1 >= 0 && t2 >= 0) { trace.t = min(t1,t2); }
    else if (t1 >= 0 && t2 < 0) { trace.t = t1; }
    else if (t1 < 0 && t2 >= 0) { trace.t = t2; }
    else { return trace; }
    let ot = trace.rd * trace.t;
    trace.hit = uni.cam_x + ot;
    trace.normal = normalize(ot + co);
    let hitclip = uni.mvp * v4(trace.hit, 1);
    trace.clip_depth = hitclip.z / hitclip.w;
    return trace;
}

@fragment fn particle_frag(input:PartIO) -> FragDepth {

    let m = mbuf[input.mesh];
    let color = m.pcolor;
    if (color.a < 0.5) { discard; }
    var rgb = color.rgb;
    if (input.selected == 1u) {
        rgb = 1 - rgb;
    }
    let trace = trace_sphere(input.vertx, input.partx, uni.r);
    if (trace.t < 0) { discard; }
    return FragDepth(frag(trace.hit, trace.normal, v4(rgb,1.0f)), trace.clip_depth);
}

@fragment fn lights_frag(input:LightIO) -> FragDepth {
    let trace = trace_sphere(input.vertx, input.lightx, input.size);
    var mag = .01/(1-dot(-trace.normal, trace.rd));
    return FragDepth(v4(input.color*mag,mag), trace.clip_depth);
}

struct GndIO {
    @builtin(position) position:v4,
    @location(0) vertx:v3,
};

const rgnd = 200.0f;
const gnd_color = v4(1, 1, 1, 1);

@vertex fn ground_vert(@builtin(vertex_index) vertidx:u32) -> GndIO {
    var out:GndIO;
    let vx = cube[vertidx];
    out.vertx = vx*rgnd;
    out.position = uni.mvp * v4(rgnd * (vx - v3(0,0,1)), 1);
    return out;
}

fn checkers(xy:v2) -> f32 {
    return f32(abs(i32(floor(xy.x)) + i32(floor(xy.y))) % 2);
}

@fragment fn ground_frag(input:GndIO) -> FragDepth {
    let trace = trace_sphere(input.vertx, v3(0,0,-rgnd), rgnd);
    if (trace.t < 0) { discard; }
    let pattern = .2*checkers(trace.hit.xy/.1) + .3*checkers(trace.hit.xy) + 0.1;
    let fade = 10/(10 + length(trace.hit.xy));
    var color = v4(v3(clamp(pattern,.2,.8)*fade),1);
    return FragDepth(frag(trace.hit, trace.normal, color), trace.clip_depth);
}
