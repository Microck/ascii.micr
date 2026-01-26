export const COMPUTE_SHADER = `
struct Params {
    img_width: f32,
    img_height: f32,
    char_width: f32,
    char_height: f32,
    grid_width: f32,
    grid_height: f32,
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var atlas_texture: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> char_grid: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let grid_x = id.x;
    let grid_y = id.y;
    
    if (grid_x >= u32(params.grid_width) || grid_y >= u32(params.grid_height)) {
        return;
    }

    let cw = u32(params.char_width);
    let ch = u32(params.char_height);
    let start_x = grid_x * cw;
    let start_y = grid_y * ch;

    var best_diff = 999999.0;
    var best_char = 0u;

    // Loop through 256 characters
    for (var c = 32u; c < 127u; c++) { // Optimization: Just ASCII 32-126 for now
        var current_diff = 0.0;
        
        let atlas_col = c % 16u;
        let atlas_row = c / 16u;
        let atlas_x = atlas_col * cw;
        let atlas_y = atlas_row * ch;

        // Compare pixel block
        for (var y = 0u; y < ch; y++) {
            for (var x = 0u; x < cw; x++) {
                let in_color = textureLoad(input_texture, vec2<i32>(i32(start_x + x), i32(start_y + y)), 0).r;
                let atlas_color = textureLoad(atlas_texture, vec2<i32>(i32(atlas_x + x), i32(atlas_y + y)), 0).r;
                let d = in_color - atlas_color;
                current_diff += d * d;
            }
        }

        if (current_diff < best_diff) {
            best_diff = current_diff;
            best_char = c;
        }
    }

    let index = grid_y * u32(params.grid_width) + grid_x;
    char_grid[index] = best_char;
}
`;

export const RENDER_SHADER = `
struct Params {
    img_width: f32,
    img_height: f32,
    char_width: f32,
    char_height: f32,
    grid_width: f32,
    grid_height: f32,
}

@group(0) @binding(0) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var atlas_texture: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> char_grid: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;

    if (x >= u32(params.img_width) || y >= u32(params.img_height)) {
        return;
    }

    let cw = u32(params.char_width);
    let ch = u32(params.char_height);
    
    let grid_x = x / cw;
    let grid_y = y / ch;

    if (grid_x >= u32(params.grid_width) || grid_y >= u32(params.grid_height)) {
        return;
    }

    let index = grid_y * u32(params.grid_width) + grid_x;
    let char_code = char_grid[index];

    let local_x = x % cw;
    let local_y = y % ch;

    let atlas_col = char_code % 16u;
    let atlas_row = char_code / 16u;
    let atlas_x = atlas_col * cw + local_x;
    let atlas_y = atlas_row * ch + local_y;

    let color = textureLoad(atlas_texture, vec2<i32>(i32(atlas_x), i32(atlas_y)), 0);
    textureStore(output_texture, vec2<i32>(i32(x), i32(y)), color);
}
`;
