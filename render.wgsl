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
    @location(0) vertpos:v3,
    @location(1) norm:v3,
    @location(2) uv:v2,
    @location(3) @interpolate(flat) mesh:u32,
};



@vertex fn vert_surface(@location(0) pos:v3,
                        @location(1) norm:v3,
                        @location(2) mesh:u32,
                        @location(3) uv:v2) -> VertOut {
    var output:VertOut;
    output.position = camera.projection * camera.modelview * v4(pos, 1.0);
    output.vertpos = pos;
    output.norm = norm;
    output.uv = uv;
    output.mesh = mesh;
    return output;
}

/*@vertex fn vert_particle(@builtin(vertex_index) vertidx:u32,
                         @builtin(instance_index) instidx:u32) -> VertOut {
    let p = &particles[instidx];
    let vpos = particle_mesh[vertidx] * camera.d/2.0;
    var output = vert((*p).si + vpos, normalize(vpos));
    output.color = meshes[(*p).mesh].pcolor;
    if (i32(instidx) == camera.selection) {
        output.color.r = 1 - output.color.r;
    }
    return output;
    }*/


@fragment fn frag_opaque(input:VertOut) -> @location(0) v4 {
    return frag_surface(input, false);
}

@fragment fn frag_trans(input:VertOut) -> @location(0) v4 {
    return frag_surface(input, true);
}


fn frag_surface(input:VertOut, transparents:bool) -> v4 {
    let m = &meshes[input.mesh];
    let color = (*m).color * select(textureSample(tex, samp, input.uv, (*m).tex), vec4(1), (*m).tex < 0);
    if (transparents && color.a >= 0.9999) { discard; }
    if (!transparents && color.a < 0.9999) { discard; }    

    
    if (color.a < 0.0001) { discard; }
    var mix = color.rgb * ambient;
    for (var i = 0; i < ${numLights}; i += 1) {
        let light = &lights[i];
        var lightdir = (*light).pos - input.vertpos;
        let distance = length(lightdir);
        let lightmag = (*light).color * (*light).power / (0.1 + (distance*distance));
        lightdir = normalize(lightdir);
        
            
        let lambertian = max(dot(lightdir, input.norm), 0.0);
        var specular = 0.0f;
        if (lambertian > 0.0) {
            let viewdir = normalize(camera.pos - input.vertpos);
            let reflectdir = reflect(-lightdir, input.norm);
            let specAngle = max(dot(reflectdir, viewdir), 0.0);
            specular = pow(specAngle, shininess);
        }
        mix += lightmag * (color.rgb*lambertian + specColor*specular);
    }
    
    return v4(mix, color.a);

}

