@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, write> result: array<f32>;

@group(0) @binding(3) var<storage, read> n: u32;
@group(0) @binding(4) var<storage, read> variables: u32;
@group(0) @binding(5) var<storage, read> features: u32;


@compute @workgroup_size(n, v, f)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    let col = id.y;
    let a_rows = 2u; // Number of rows in matrix A
    let a_cols = 3u; // Number of columns in matrix A
    let b_cols = 2u; // Number of columns in matrix B

    var sum: f32 = 0.0;
    for (var i = 0u; i < a_cols; i = i + 1u) {
        sum = sum + a[row * a_cols + i] * b[i * b_cols + col];
    }

    result[row * b_cols + col] = sum;
}
