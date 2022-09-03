type v2 = vec2<f32>;
type v4 = vec4<f32>;
type v3 = vec3<f32>;
type m3 = mat3x3<f32>;

const lightpow = 20.0;
const shininess = 100.0;
const ambient = 0.02f;
const particle_mesh = array<v3, ${particleDraws}>(${particleMesh});
const I3 = m3(1,0,0,0,1,0,0,0,1);

struct VertOut {
    @builtin(position) position: v4,
    @location(1) vertpos:v3,
    @location(2) normal:v3,
    @location(3) uv:v2,
    @location(4) color:v4,
    @location(5) @interpolate(flat) tex:i32
};

fn vert(position:v3, normal:v3) -> VertOut {
    var output:VertOut;
    output.position = camera.projection * camera.modelview * v4(position, 1.0);
    output.vertpos = position;
    output.tex = -1;
    output.normal = normal;
    return output;
}

@vertex fn vert_surface(@location(0) vidx:u32,
                        @location(1) uv:v2) -> VertOut {
    let v = &vertices[vidx];
    let m = &meshes[(*v).mesh];
    var output = vert((*v).pos, (*v).norm);
    output.uv = uv;
    output.tex = (*m).tex;
    output.color = (*m).color;
    return output;
}

@vertex fn vert_particle(@builtin(vertex_index) vertidx:u32,
                         @builtin(instance_index) instidx:u32) -> VertOut {
    let p = &particles[instidx];
    let vpos = particle_mesh[vertidx] * camera.d/2.0;
    var output = vert((*p).si + vpos, normalize(vpos));
    output.color = meshes[(*p).mesh].pcolor;
    if (i32(instidx) == camera.selection) {
        output.color.r = 1 - output.color.r;
    }
    return output;
}

@fragment fn frag(input:VertOut) -> @location(0) v4 {
    let sample = select(textureSample(tex, samp, input.uv, input.tex),vec4(1), input.tex < 0);
    var color = input.color * sample;
    if (color.a == 0) { discard; }
    var lightColor = v3(0);
    for (var i = 0; i < ${numLights}; i += 1) {
        let light = &lights[i];
        var lightdir = (*light).pos - input.vertpos;
        var distance = length(lightdir);
        distance = distance * distance;
        let lightmag = (*light).power / ((*light).power + distance);
        lightdir = normalize(lightdir);
        let lambertian = lightmag * max(dot(lightdir, input.normal), 0.00001);
        let viewdir = normalize(camera.pos - input.vertpos);
        let halfdir = normalize(viewdir + lightdir);
        let specular = lightmag * pow(max(dot(halfdir, input.normal), 0.0), shininess);
        lightColor += (*light).color * (lambertian + specular);
    }
    return v4(color.rgb * (ambient + lightColor), color.a);

}

