type v2 = vec2<f32>;
type v4 = vec4<f32>;
type v3 = vec3<f32>;
type m3 = mat3x3<f32>;
type m4 = mat4x4<f32>;

const shininess = 16.0;
const ambient = 0.2f;


const tetrahedron = array<v3,8>(v3( 1, 1, 1), v3( 1,-1,-1), v3(-1, 1,-1), v3(-1,-1, 1),
                                v3( 1, 1, 1), v3(-1,-1, 1), v3( 1,-1,-1), v3(-1, 1,-1));

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
            let viewdir = normalize(camera.pos - worldpos);
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
    out.position = camera.projection * camera.modelview * v4(out.worldpos, 1.0);
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
    if ((*m).flags == 1) { discard; }
    var color = (*m).color * select(textureSample(tex, samp, input.uv, (*m).tex), v4(1), (*m).tex < 0);
    if (color.a < 0.0001) { discard; }
    color = frag(input.worldpos, input.norm, color);
    return v4(color.rgb, color.a);
}


@vertex fn axes_vert(@builtin(vertex_index) vertidx:u32,
                     @builtin(instance_index) instidx:u32) -> @builtin(position) v4 {
    var worldPos = v3(0);
    worldPos[instidx] = f32(2*i32(vertidx) - 1);
    return camera.projection * camera.modelview * v4(worldPos, 1.0);
}

@fragment fn axes_frags() -> @location(0) v4 {
    return v4(1,1,1,1);
}

@vertex fn normals_vert(@builtin(vertex_index) vertidx:u32,
                     @location(0) vertPos:v3,
                     @location(1) norm:v3) -> @builtin(position) v4 {
    var worldPos = vertPos;
    if (vertidx == 1u) {
        worldPos += norm * camera.r * 2;
    }    
    return camera.projection * camera.modelview * v4(worldPos, 1.0);
}

@fragment fn normals_frag() -> @location(0) v4 {
    return v4(1,1,1,1);
}


struct PartIO {
    @builtin(position) position:v4,
    @location(0) partpos:v3,
    @location(1) vertpos:v3,
    @location(2) @interpolate(flat) mesh:u32,
    @location(3) @interpolate(flat) selected:u32,
};

@vertex fn particle_vert(@builtin(vertex_index) vertidx:u32,
                     @builtin(instance_index) instidx:u32,
                     @location(0) partpos:v3,
                     @location(1) mesh:u32) -> PartIO {

    var out:PartIO;
    out.partpos = partpos;
    out.vertpos = tetrahedron[vertidx] * sqrt(3) * camera.r;
    out.position = camera.projection * camera.modelview * v4(out.partpos + out.vertpos,1);
    out.mesh = mesh;
    out.selected = select(0u, 1u, i32(instidx) == camera.selection);
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
    trace.rd = normalize(vertpos + center - camera.pos);
    let co = camera.pos - center;
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
    trace.hit = camera.pos + ot;
    trace.normal = normalize(ot + co);
    let hitclip = camera.projection * camera.modelview * v4(trace.hit, 1);
    trace.clip_depth = hitclip.z / hitclip.w;
    return trace;
}

@fragment fn particle_frag(input:PartIO) -> FragDepth {
    let m = &meshes[input.mesh];
    if ((*m).flags == 1) { discard; }
    let color = (*m).pcolor;
    if (color.a < 0.5) { discard; }
    var rgb = color.rgb;
    if (input.selected == 1u) {
        rgb = 1 - rgb;
    }
    let trace = trace_sphere(input.vertpos, input.partpos, camera.r);
    return FragDepth(frag(trace.hit, trace.normal, v4(rgb,1.0f)), trace.clip_depth);   
}


struct LightIO {
    @builtin(position) position:v4,
    @location(0) lightpos:v3,
    @location(1) vertpos:v3,
    @location(2) color:v3,
    @location(3) size:f32,
};

@vertex fn lights_vert(@builtin(vertex_index) vertidx:u32,
                      @builtin(instance_index) instidx:u32) -> LightIO {   
    let l = &lights[instidx];
    var out:LightIO;

    out.lightpos = (*l).pos;
    out.size = .4*sqrt((*l).power);
    let vpos = cube[vertidx];
    out.vertpos = vpos * out.size;
    out.position = camera.projection * camera.modelview * v4(out.lightpos + out.vertpos * out.size,1);
    out.color = (*l).color;
    return out;
}
        

@fragment fn lights_frag(input:LightIO) -> FragDepth {
    let trace = trace_sphere(input.vertpos, input.lightpos, input.size);
    let mag = pow(dot(-trace.normal, trace.rd), 20);
    return FragDepth(v4(input.color * mag, 1), trace.clip_depth);
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
    out.position = camera.projection * camera.modelview * v4(rgnd * (vpos - v3(0,0,1)), 1);
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



