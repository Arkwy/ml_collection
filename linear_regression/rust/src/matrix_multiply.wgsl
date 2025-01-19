@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    let a_rows = 2u; // Number of rows in matrix A
    let a_cols = 3u; // Number of columns in matrix A
    let b_cols = 2u; // Number of columns in matrix B

    var sum: f32 = 0.0;
    for (var i = 0u; i < a_cols; i = i + 1u) {
        sum = sum + a[row * a_cols + i] * b[i * b_cols + col];
    }

    result[row * b_cols + col] = sum;
}
