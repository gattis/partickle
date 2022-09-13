type v2 = vec2<f32>;
type v4 = vec4<f32>;
type v3 = vec3<f32>;
type m3 = mat3x3<f32>;
type m4 = mat4x4<f32>;

const specColor = v3(1.0f,1.0f,1.0f);
const shininess = 4.0;
const ambient = 0.5f;
const particle_mesh = array<v3, ${partDraws}>(${partWgsl});
const decalR = 0.17321;
const decal = array<v2, 3>(v2(-.3,-.17321), v2(0,0.34641), v2(.3,-.17321));


struct IO {
    @builtin(position) position: v4,
    @location(0) worldpos:v3,
    @location(1) norm:v3,
    @location(2) uv:v2,
    @location(3) @interpolate(flat) selected:u32,
    @location(4) @interpolate(flat) mesh:u32,
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

@vertex fn vert_part(@builtin(vertex_index) vertidx:u32,
                     @builtin(instance_index) instidx:u32,
                     @location(0) partPos:v3,
                     @location(1) mesh:u32) -> IO {
    var out:IO;
    let vertPos = particle_mesh[vertidx];
    out.worldpos = partPos + vertPos;
    out.position = camera.projection * camera.modelview * v4(out.worldpos, 1.0);
    out.norm = normalize(vertPos);
    out.mesh = mesh;
    out.selected = select(0u, 1u, i32(instidx) == camera.selection);
    return out;
}

fn frag(input:IO, color:v4) -> v4 {
    var mix = color.rgb * ambient;
    for (var i = 0; i < ${numLights}; i += 1) {
        let light = &lights[i];
        var lightdir = (*light).pos - input.worldpos;
        let distance = length(lightdir);
        let lightmag = (*light).color * (*light).power / (0.1 + (distance*distance));
        lightdir = normalize(lightdir);
        
            
        let lambertian = max(dot(lightdir, input.norm), 0.0);
        var specular = 0.0f;
        if (lambertian > 0.0) {
            let viewdir = normalize(camera.pos - input.worldpos);
            let reflectdir = reflect(-lightdir, input.norm);
            let specAngle = max(dot(reflectdir, viewdir), 0.0);
            specular = pow(specAngle, shininess);
        }
        mix += lightmag * (color.rgb*lambertian + specColor*specular);
    }
    
    return v4(mix, color.a);

}


@fragment fn frag_part(input:IO) -> @location(0) v4 {
    var color = meshes[input.mesh].pcolor;
    if (color.a < 0.5) { discard; }
    color.a = 1.0;
    if (input.selected == 1u) {
        color.r = 1 - color.r;
    }
    return frag(input, color);
}

fn frag_surf(input:IO, transp:bool) -> v4 {
    let m = &meshes[input.mesh];
    let color = (*m).color * select(textureSample(tex, samp, input.uv, (*m).tex), vec4(1), (*m).tex < 0);
    if (transp && color.a >= 0.9999) { discard; }
    if (!transp && color.a < 0.9999) { discard; }
    if (color.a < 0.0001) { discard; }
    return frag(input, color);
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
        worldPos += norm * camera.d;
    }    
    return camera.projection * camera.modelview * v4(worldPos, 1.0);
}

@fragment fn frag_norm() -> @location(0) v4 {
    return v4(1,1,1,1);
}

struct LightIO {
    @builtin(position) position:v4,
    @location(1) vertpos:v2,
    @location(2) color:v3
};

@vertex fn vert_light(@builtin(vertex_index) vertidx:u32,
                      @builtin(instance_index) instidx:u32) -> LightIO {
    let l = &lights[instidx];
    let vp = decal[vertidx];
    let mv = transpose(camera.modelview);
    let pos = (*l).pos + mv[0].xyz*vp.x + mv[1].xyz*vp.y;
    var out:LightIO;
    out.position = camera.projection * camera.modelview * v4(pos, 1);
    out.vertpos = decal[vertidx];
    out.color = (*l).color + v3(.5,.5,.5);
    return out;
}

@fragment fn frag_light(input:LightIO) -> @location(0) v4 {
    let r = length(input.vertpos);
    if (r > decalR) { discard; }
    let mag = exp(-8*r/decalR);
    return v4(input.color*mag,mag);
}
