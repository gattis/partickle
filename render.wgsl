type v2 = vec2<f32>;
type v4 = vec4<f32>;
type v3 = vec3<f32>;
type m3 = mat3x3<f32>;

const specColor = v3(1,1,1);
const shininess = 4.0;
const ambient = 0.1f;
const particle_mesh = array<v3, ${partDraws}>(${partWgsl});

struct VertOut {
    @builtin(position) position: v4,
    @location(0) worldpos:v3,
    @location(1) norm:v3,
    @location(2) uv:v2,
    @location(3) @interpolate(flat) selected:u32,
    @location(4) @interpolate(flat) mesh:u32,
};

struct FragIn {
    @builtin(front_facing) front: bool,
    @location(0) worldpos:v3,
    @location(1) norm:v3,
    @location(2) uv:v2,
    @location(3) @interpolate(flat) selected:u32,
    @location(4) @interpolate(flat) mesh:u32,
}



@vertex fn vert_surf(@location(0) pos:v3,
                     @location(1) norm:v3,
                     @location(2) mesh:u32,
                     @location(3) uv:v2) -> VertOut {
    var output:VertOut;
    output.worldpos = pos;
    output.position = camera.projection * camera.modelview * v4(output.worldpos, 1.0);
    output.norm = norm;
    output.uv = uv;
    output.mesh = mesh;
    return output;
}

@vertex fn vert_part(@builtin(vertex_index) vertidx:u32,
                     @builtin(instance_index) instidx:u32,
                     @location(0) partPos:v3,
                     @location(1) mesh:u32) -> VertOut {
    var output:VertOut;
    let vertPos = particle_mesh[vertidx];
    output.worldpos = partPos + vertPos;
    output.position = camera.projection * camera.modelview * v4(output.worldpos, 1.0);
    output.norm = normalize(vertPos);
    output.mesh = mesh;
    output.selected = select(0u, 1u, i32(instidx) == camera.selection);
    return output;
}

@vertex fn vert_norm(@builtin(vertex_index) vertidx:u32,
                     @location(0) vertPos:v3,
                     @location(1) norm:v3) -> @builtin(position) v4 {

    var worldPos = vertPos;
    if (vertidx == 1u) {
        worldPos += norm * 0.05;
    }
    
    return camera.projection * camera.modelview * v4(worldPos, 1.0);

}

@fragment fn frag_norm() -> @location(0) v4 {
    return v4(1,1,1,1);
}

@fragment fn frag_part(input:FragIn) -> @location(0) v4 {
    var color = meshes[input.mesh].pcolor;
    if (color.a < 0.5) { discard; }
    color.a = 1.0;
    if (input.selected == 1u) {
        color.r = 1 - color.r;
    }
    return frag(input, color);
}


@fragment fn frag_surf_opaque(input:FragIn) -> @location(0) v4 {
    return frag_surf(input, false);
}

@fragment fn frag_surf_transp(input:FragIn) -> @location(0) v4 {
    return frag_surf(input, true);
}

fn frag_surf(input:FragIn, transp:bool) -> v4 {
    let m = &meshes[input.mesh];
    let color = (*m).color * select(textureSample(tex, samp, input.uv, (*m).tex), vec4(1), (*m).tex < 0);
    if (transp && color.a >= 0.9999) { discard; }
    if (!transp && color.a < 0.9999) { discard; }
    if (color.a < 0.0001) { discard; }
    return frag(input, color);
}


fn frag(input:FragIn, color:v4) -> v4 {
    if (!input.front) { discard; }
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

