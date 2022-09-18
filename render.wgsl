type v2 = vec2<f32>;
type v4 = vec4<f32>;
type v3 = vec3<f32>;
type m3 = mat3x3<f32>;
type m4 = mat4x4<f32>;

const shininess = 4.0;
const ambient = 0.2f;



const tetrahedron = array<v3,12>(v3( 1, 1, 1), v3( 1,-1,-1), v3(-1, 1,-1),
                                 v3( 1, 1, 1), v3(-1, 1,-1), v3(-1,-1, 1),
                                 v3( 1, 1, 1), v3(-1,-1, 1), v3( 1,-1,-1),
                                 v3( 1,-1,-1), v3(-1,-1, 1), v3(-1, 1,-1));

const cube = array<v3,36>(v3(-1, -1, 1), v3(-1, 1, -1), v3(-1, -1, -1),
                          v3(-1, 1, 1), v3(1, 1, -1), v3(-1, 1, -1),
                          v3(1, 1, 1), v3(1, -1, -1), v3(1, 1, -1),
                          v3(1, -1, 1), v3(-1, -1, -1), v3(1, -1, -1),
                          v3(1, 1, -1), v3(-1, -1, -1), v3(-1, 1, -1),
                          v3(-1, 1, 1), v3(1, -1, 1), v3(1, 1, 1),
                          v3(-1, -1, 1), v3(-1, 1, 1), v3(-1, 1, -1),
                          v3(-1, 1, 1), v3(1, 1, 1), v3(1, 1, -1),
                          v3(1, 1, 1), v3(1, -1, 1), v3(1, -1, -1),
                          v3(1, -1, 1), v3(-1, -1, 1), v3(-1, -1, -1),
                          v3(1, 1, -1), v3(1, -1, -1), v3(-1, -1, -1),
                          v3(-1, 1, 1), v3(-1, -1, 1), v3(1, -1, 1));


const bb = array<v2,3>(v2(-1.7321,-1), v2(0,2), v2(1.7321,-1));

struct IO {
    @builtin(position) position: v4,
    @location(0) worldpos:v3,
    @location(1) norm:v3,
    @location(2) uv:v2,
    @location(3) @interpolate(flat) mesh:u32,
};


@vertex fn vert_surf(@location(0) pos:v3,
                     @location(1) norm:v3,
                     @location(2) mesh:u32,
                     @location(3) uv:v2) -> IO {
    var out:IO;
    out.worldpos = pos;
    out.position = camera.projection * camera.modelview * v4(out.worldpos, 1.0);
    out.norm = norm;
    out.uv = uv;
    out.mesh = mesh;
    return out;
}


fn frag(worldpos:v3, norm:v3, color:v4) -> v4 {
    var mix = color.rgb * ambient;
    for (var i = 0; i < ${numLights}; i += 1) {
        let light = &lights[i];
        var lightdir = (*light).pos - worldpos;
        let distance = length(lightdir);
        let lightmag = (*light).color * (*light).power / (0.5 + (distance*distance));
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
    
    return v4(mix, color.a);

}

fn frag_surf(input:IO, transp:bool) -> v4 {
    let m = &meshes[input.mesh];
    let color = (*m).color * select(textureSample(tex, samp, input.uv, (*m).tex), vec4(1), (*m).tex < 0);
    if (transp && color.a >= 0.9999) { discard; }
    if (!transp && color.a < 0.9999) { discard; }
    if (color.a < 0.0001) { discard; }
    return frag(input.worldpos, input.norm, color);
}

@fragment fn frag_surf_opaque(input:IO) -> @location(0) v4 {
    return frag_surf(input, false);
}

@fragment fn frag_surf_transp(input:IO) -> @location(0) v4 {
    return frag_surf(input, true);
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


struct FragDepth {
    @location(0) color: v4,
    @builtin(frag_depth) depth:f32
};

@fragment fn frag_part(input:PartIO) -> FragDepth {

    let color = meshes[input.mesh].pcolor;
    if (color.a < 0.5) { discard; }
    var rgb = color.rgb;
    if (input.selected == 1u) {
        rgb = 1 - rgb;
    }

    let rd = normalize(input.partpos + input.vertpos - camera.pos);
    let oc = camera.pos - input.partpos;
    let b = dot(oc,rd);
    let c = b*b - dot(oc,oc) + camera.r*camera.r;
    if (c < 0) { discard; }
    let t = -b - sqrt(c);
    if (t < 0) { discard; }
    var worldpos = camera.pos + rd*t;
    let clippos = camera.projection * camera.modelview * v4(worldpos,1);
    let normal = normalize(worldpos - input.partpos);
    return FragDepth(frag(worldpos, normalize(input.vertpos), v4(rgb,1.0f)), clippos.z/clippos.w);
   
}


struct LightIO {
    @builtin(position) position:v4,
    @location(0) lightpos:v3,
    @location(1) vertpos:v3,
    @location(2) color:v3,
    @location(3) size:f32,
};

@vertex fn vert_light(@builtin(vertex_index) vertidx:u32,
                      @builtin(instance_index) instidx:u32) -> LightIO {   
    let l = &lights[instidx];
    var out:LightIO;

    out.lightpos = (*l).pos;
    out.size = 0.08f * sqrt((*l).power);
    out.vertpos = tetrahedron[vertidx] * sqrt(3) * out.size;
    out.position = camera.projection * camera.modelview * v4(out.lightpos + out.vertpos,1);
    out.color = (*l).color;
    return out;
}

@fragment fn frag_light(input:LightIO) -> FragDepth {
    let rd = normalize(input.lightpos + input.vertpos - camera.pos);
    let oc = camera.pos - input.lightpos;
    let b = dot(oc,rd);
    let c = b*b - dot(oc,oc);
    let cs = c + input.size*input.size;
    if (cs < 0) { discard; }
    let t = -b - sqrt(cs);
    if (t < 0) { discard; }
    var worldpos = camera.pos + rd*t;
    let clippos = camera.projection * camera.modelview * v4(worldpos,1);
    let vertpos = worldpos - input.lightpos;
    let mag = pow(35*abs(cs),6);
    return FragDepth(v4(input.color*mag,mag), clippos.z/clippos.w);

}




