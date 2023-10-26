alias v2 = vec2<f32>;
alias v2i = vec2<i32>;
alias v4 = vec4<f32>;
alias v3 = vec3<f32>;
alias m3 = mat3x3<f32>;
alias m4 = mat4x4<f32>;

const shininess = 4.0;
const ambient = 0.03f;




fn shadow(x:v3, i:i32) -> f32 {
    let l = lbuf[i];
    let ss = l.viewproj * v4(x,1);
    let c = ss.xyz/ss.w * v3(.5, -.5, 1) + v3(.5,.5, -1e-6);
    if (ss.z < 0 || c.x < 0 || c.y < 0 || c.x > 1 || c.y > 1) { return 0; }
    if (l.shadow == 0) { return 1.0; }
    let uv = c.xy;
    let d = c.z;
    return clamp((textureSampleCompareLevel(shadowMaps, shadowSamp, uv, i, d, v2i( 0, 0)) +
                  textureSampleCompareLevel(shadowMaps, shadowSamp, uv, i, d, v2i(-1,-1)) +
                  textureSampleCompareLevel(shadowMaps, shadowSamp, uv, i, d, v2i(-1, 0)) +
                  textureSampleCompareLevel(shadowMaps, shadowSamp, uv, i, d, v2i(-1, 1)) +
                  textureSampleCompareLevel(shadowMaps, shadowSamp, uv, i, d, v2i( 0,-1)) +
                  textureSampleCompareLevel(shadowMaps, shadowSamp, uv, i, d, v2i( 0, 1)) +
                  textureSampleCompareLevel(shadowMaps, shadowSamp, uv, i, d, v2i( 1,-1)) +
                  textureSampleCompareLevel(shadowMaps, shadowSamp, uv, i, d, v2i( 1, 0)) +
                  textureSampleCompareLevel(shadowMaps, shadowSamp, uv, i, d, v2i( 1, 1)))/9.0, 0, 1);
}

fn frag(worldx:v3, norm:v3, color:v4) -> v4 {
    var mix = color.rgb * ambient;
    for (var i = 0; i < ${numLights}; i += 1) {
        let l = lbuf[i];
        let ray = l.x - worldx;
        let dir = normalize(ray);
        let power = .5 * l.power / dot(ray,ray);
        let lambert = clamp(dot(dir, norm), 0, 1);
        var specular = 0.0f;
        if (lambert > 0.0) {
            let viewdir = normalize(eye.x - worldx);
            let specAngle = clamp(dot(reflect(-dir, norm), viewdir), 0, 1);
            specular = pow(specAngle, shininess);
        }
        let s = shadow(worldx, i);
        mix += l.color * power * s * (color.rgb*lambert + specular);
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
    out.position = eye.viewproj * v4(out.worldx, 1.0);
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

struct Depth {
    @builtin(frag_depth) depth:f32
};

@fragment fn surface_frag(input:SurfIn) -> @location(0) v4 {
    let m = mbuf[input.mesh];
    if (m.fluid == 1) { discard; }
    var color = m.color * select(textureSample(tex, texSamp, input.uv, m.tex), v4(1), m.tex < 0);
    if (color.a < 0.0001) { discard; }
    color = frag(input.worldx, input.norm, color);
    return v4(color.rgb, color.a);
}

@vertex fn vnormals_vert(@builtin(vertex_index) vertidx:u32,
                         @location(0) vertx:v3,
                         @location(1) norm:v3) -> @builtin(position) v4 {
    var worldx = vertx;
    if (vertidx == 1u) {
        worldx += norm * 0.05;
    }
    return eye.viewproj * v4(worldx, 1.0);
}

@fragment fn vnormals_frag() -> @location(0) v4 {
    return v4(.2,1,.1,1);
}

@vertex fn velocity_vert(@builtin(vertex_index) vertidx:u32,
                         @location(0) partx:v3,
                         @location(1) partv:v3) -> @builtin(position) v4 {
    var worldx = partx;
    if (vertidx == 1u) {
        worldx += partv;
    }
    return eye.viewproj * v4(worldx, 1.0);
}

@fragment fn velocity_frag() -> @location(0) v4 {
    return v4(1,.2,.1,1);
}




fn mul3(m:m4, v:v3) -> v3 {
    let mv = m*v4(v,1);
    return mv.xyz / mv.w;
}


const sq3 = 1.73205077648;


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
    trace.rd = normalize(vertx + center - eye.x);
    let co = eye.x - center;
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
    trace.hit = eye.x + ot;
    trace.normal = normalize(ot + co);
    let hitclip = eye.viewproj * v4(trace.hit, 1);
    trace.clip_depth = hitclip.z / hitclip.w;
    return trace;
}

fn impostor(vertidx:u32, x:v3, r:f32) -> v3 {
    let c = length(x - eye.x);
    let rxp = r*c/sqrt(c*c-r*r);    
    let fwd = normalize(x - eye.x);
    var right = v3(-fwd.y, fwd.x, 0);
    var lright = length(right);
    right = select(right/lright, v3(0,1,0), lright == 0);
    let up = normalize(cross(fwd,right));
    return array(rxp*(2*up), rxp*(sq3*right - up), rxp*(-sq3*right - up))[vertidx];
}


struct PartIO {
    @builtin(position) position:v4,
    @location(0) @interpolate(flat) partx:v3,
    @location(1) vertx:v3,
    @location(2) @interpolate(flat) mesh:u32,
    @location(3) @interpolate(flat) selected:u32,
};

@vertex fn particle_vert(@builtin(vertex_index) vertidx:u32,
                         @builtin(instance_index) instidx:u32,
                         @location(0) partx:v3,
                         @location(1) mesh:u32) -> PartIO {
    var out:PartIO;
    let imp = impostor(vertidx, partx, u.r);
    out.partx = partx;
    out.vertx = imp;
    out.position = eye.viewproj * v4(out.partx + out.vertx,1);
    out.mesh = mesh;
    out.selected = select(0u, 1u, i32(instidx) == u.selection);
    return out;
}

@fragment fn particle_frag(input:PartIO) -> FragDepth {
    let m = mbuf[input.mesh];
    let color = m.pcolor;
    if (color.a < 0.5) { discard; }
    var rgb = color.rgb;
    if (input.selected == 1u) {
        rgb = 1 - rgb;
    }
    let trace = trace_sphere(input.vertx, input.partx, u.r);
    if (trace.t < 0) { discard; }
    return FragDepth(frag(trace.hit, trace.normal, v4(rgb,1.0f)), trace.clip_depth);
}



@fragment fn particle_depth(input:PartIO) -> Depth {
    let trace = trace_sphere(input.vertx, input.partx, u.r);
    if (trace.t < 0) { return Depth(1.0); }
    return Depth(trace.clip_depth);
}


struct LightIO {
    @builtin(position) position:v4,
    @location(0) @interpolate(flat) lightx:v3,
    @location(1) vertx:v3,
    @location(2) @interpolate(flat) color:v3,
    @location(3) @interpolate(flat) size:f32,
};

@vertex fn lights_vert(@builtin(vertex_index) vertidx:u32,
                       @builtin(instance_index) instidx:u32) -> LightIO {
    let l = lbuf[instidx];
    var out:LightIO;
    out.lightx = l.x;
    out.size = .02*sqrt(l.power);
    let imp = impostor(vertidx, out.lightx, out.size);
    out.vertx = imp;
    out.position = eye.viewproj * v4(out.lightx + out.vertx,1);
    out.color = l.color;
    return out;
}

@fragment fn lights_frag(input:LightIO) -> FragDepth {
    let trace = trace_sphere(input.vertx, input.lightx, input.size);
    if (trace.t < 0) { discard; }
    var d = clamp(length(input.vertx)/input.size, 0, 1);
    d = 6*pow(d,5) - 15*pow(d,4) + 10*pow(d,3);
    d = 1 - d;   
    let c = d*v4(input.color, 1);    
    return FragDepth(c, trace.clip_depth);
}

struct WallIO {
    @builtin(position) position:v4,
    @location(0) worldx:v3,
    @location(1) @interpolate(flat) plane:u32,
};


@vertex fn walls_vert(@builtin(vertex_index) i:u32) -> WallIO {
    var out:WallIO;
    let p = u.spacemax;
    let n = u.spacemin;    
    out.worldx = array(v3(n.x,p.y,p.z),v3(n.x,n.y,n.z),v3(n.x,n.y,p.z),
                       v3(n.x,p.y,p.z),v3(n.x,p.y,n.z),v3(n.x,n.y,n.z),
                       v3(p.x,p.y,p.z),v3(p.x,n.y,p.z),v3(p.x,p.y,n.z),
                       v3(p.x,p.y,n.z),v3(p.x,n.y,p.z),v3(p.x,n.y,n.z),
                       v3(p.x,n.y,n.z),v3(n.x,n.y,p.z),v3(n.x,n.y,n.z),
                       v3(p.x,n.y,n.z),v3(p.x,n.y,p.z),v3(n.x,n.y,p.z),
                       v3(n.x,p.y,n.z),v3(p.x,p.y,p.z),v3(p.x,p.y,n.z),
                       v3(n.x,p.y,n.z),v3(n.x,p.y,p.z),v3(p.x,p.y,p.z),
                       v3(n.x,p.y,n.z),v3(p.x,n.y,n.z),v3(n.x,n.y,n.z),
                       v3(n.x,p.y,n.z),v3(p.x,p.y,n.z),v3(p.x,n.y,n.z),
                       v3(p.x,p.y,p.z),v3(n.x,n.y,p.z),v3(p.x,n.y,p.z),
                       v3(p.x,p.y,p.z),v3(n.x,p.y,p.z),v3(n.x,n.y,p.z))[i];
    out.plane = i / 6; 
    out.position = eye.viewproj * v4(out.worldx, 1);
    return out;
}


const pi = 3.141592653589793;

fn grid(v:v2, d:f32, a:f32) -> f32 {
    let p = 1/(a*d);
    let sx = tanh(p*sin(pi*v.x*d));
    let sy = tanh(p*sin(pi*v.y*d));
    return min(pow(sx*sx,p),pow(sy*sy,p));
}


@fragment fn walls_frag(input:WallIO) -> @location(0) v4 {
    let w = input.worldx;
    let p = u.spacemax;
    let n = u.spacemin;

    let plane = array(w.yz, w.yz, w.xz, w.xz, w.xy, w.xy)[input.plane];
    let norm = array(v3(1,0,0),v3(-1,0,0),v3(0,1,0),v3(0,-1,0),v3(0,0,1),v3(0,0,-1))[input.plane];
    let pattern = grid(plane, 10, .004);
    let color = v4(v3(1,.9,1)*pattern, 1);
    return frag(input.worldx, norm, color);
}








    
    

