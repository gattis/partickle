type v2 = vec2<f32>;
type v4 = vec4<f32>;
type v3 = vec3<f32>;
type m3 = mat3x3<f32>;
type m4 = mat4x4<f32>;

const shininess = 64.0;
const ambient = 0.2f;


const tetrahedron = array<v3,8>(v3( 1, 1, 1), v3( 1,-1,-1), v3(-1, 1,-1), v3(-1,-1, 1),
                                v3( 1, 1, 1), v3(-1,-1, 1), v3( 1,-1,-1), v3(-1, 1,-1));
const cube = array<v3,14>(v3(-1,1,-1), v3(1,1,-1), v3(-1,-1,-1), v3(1,-1,-1), v3(1,-1,1),
                         v3(1,1,-1), v3(1,1,1), v3(-1,1,-1), v3(-1,1,1), v3(-1,-1,-1),
                         v3(-1,-1,1), v3(1,-1,1), v3(-1,1,1), v3(1,1,1));




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

@vertex fn vert_surf(@location(0) pos:v3,
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

fn ranger(x:f32, inlo:f32, inhi:f32, outlo:f32, outhi:f32) -> f32 {
    return (outhi - outlo) * (x - inlo) / (inhi - inlo) + outlo;
}


@fragment fn frag_surf(input:SurfIn) -> @location(0) v4 {
    let m = &meshes[input.mesh];
    var color = (*m).color * select(textureSample(tex, samp, input.uv, (*m).tex), vec4(1), (*m).tex < 0);
    color = frag(input.worldpos, input.norm, color);
    return v4(color.rgb, color.a);
}


@vertex fn vert_axis(@builtin(vertex_index) vertidx:u32,
                     @builtin(instance_index) instidx:u32) -> @builtin(position) v4 {
    var worldPos = v3(0);
    worldPos[instidx] = f32(2*i32(vertidx) - 1);
    return camera.projection * camera.modelview * v4(worldPos, 1.0);
}

@fragment fn frag_axis() -> @location(0) v4 {
    return v4(1,1,1,1);
}

@vertex fn vert_norm(@builtin(vertex_index) vertidx:u32,
                     @location(0) vertPos:v3,
                     @location(1) norm:v3) -> @builtin(position) v4 {
    var worldPos = vertPos;
    if (vertidx == 1u) {
        worldPos += norm * camera.r * 2;
    }    
    return camera.projection * camera.modelview * v4(worldPos, 1.0);
}

@fragment fn frag_norm() -> @location(0) v4 {
    return v4(1,1,1,1);
}


struct PartIO {
    @builtin(position) position:v4,
    @location(0) partpos:v3,
    @location(1) vertpos:v3,
    @location(2) @interpolate(flat) mesh:u32,
    @location(3) @interpolate(flat) selected:u32,
};

@vertex fn vert_part(@builtin(vertex_index) vertidx:u32,
                     @builtin(instance_index) instidx:u32,
                     @location(0) partpos:v3,
                     @location(1) mesh:u32) -> PartIO {

    var out:PartIO;
    out.partpos = partpos;
    out.vertpos = tetrahedron[vertidx] * sqrt(3) * camera.r;
    out.position = camera.projection * camera.modelview * v4(out.partpos + out.vertpos,1);
    out.mesh = mesh;
    out.selected = select(1u, 0u, i32(instidx) == camera.selection);
    return out;
}



@fragment fn frag_part(input:PartIO) -> FragDepth {

    let color = meshes[input.mesh].pcolor;
    if (color.a < 0.5) { discard; }
    var rgb = color.rgb;
    if (input.selected == 1u) {
        rgb = 1 - rgb;
    }
    let pc = camera.pos;
    let pl = input.partpos;
        
    let rd = normalize(pl + input.vertpos - pc);
    let vcl = pc - pl;
    let b = dot(vcl,rd);
    let c = b*b - dot(vcl,vcl) + camera.r*camera.r;
    if (c < 0) { discard; }
    let t = -b - sqrt(c);
    if (t < 0) { discard; }
    var pt = pc + rd*t;
    let clippos = camera.projection * camera.modelview * v4(pt,1);
    let normal = normalize(pt - pl);
    return FragDepth(frag(pt, normalize(input.vertpos), v4(rgb,1.0f)), clippos.z/clippos.w);
   
}


struct LightOut {
    @builtin(position) position:v4,
    @location(0) lightpos:v3,
    @location(1) vertpos:v3,
    @location(2) color:v3,
    @location(3) size:f32,
};

@vertex fn vert_light(@builtin(vertex_index) vertidx:u32,
                      @builtin(instance_index) instidx:u32) -> LightOut {   
    let l = &lights[instidx];
    var out:LightOut;

    out.lightpos = (*l).pos;
    out.size = 0.08f * sqrt((*l).power);
    out.vertpos = cube[vertidx] * sqrt(3) * out.size;
    out.position = camera.projection * camera.modelview * v4(out.lightpos + out.vertpos,1);
    out.color = (*l).color;
    return out;
}


struct LightIn {
    @builtin(position) position:v4,
    @location(0) lightpos:v3,
    @location(1) vertpos:v3,
    @location(2) color:v3,
    @location(3) size:f32,
};


@fragment fn frag_light(input:LightIn) -> FragDepth {
    let pc = camera.pos;
    let pl = input.lightpos;
    let vctnorm = normalize(pl + input.vertpos - pc);
    let vlc = pc - pl;
    let b = dot(vlc,vctnorm);
    let c = b*b - dot(vlc,vlc) + input.size*input.size;
    if (c < 0) { discard; }
    let t = -b - sqrt(c);
    if (t < 0) { discard; }
    let vct = vctnorm*t;
    let pt = pc + vct;
    let vtl = -vct + -vlc;
    
    let ptclip = camera.projection * camera.modelview * v4(pt,1);   
    let mag = pow(dot(normalize(vtl),vctnorm), 15);
    return FragDepth(v4(input.color*mag,1), ptclip.z/ptclip.w);

}


