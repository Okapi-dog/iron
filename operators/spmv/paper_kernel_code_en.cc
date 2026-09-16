// Single-core kernel pseudocode for sparse matrix-vector multiplication (SpMV).
// Computes y = A * x.
//
// A: sparse matrix in ELLPACK format
// A_col: column-index array
// A_val: nonzero-element array
// x: input vector
// y: output vector
// ell_width: maximum number of nonzero elements per row
// Rows: number of rows processed by one core
for (uint32_t i = 0; i < Rows; i++) {
    // Initialize the accumulator for the partial sum.
    vector_acc = vector_zeros<float32, 32>();

    // Pointers to the column indices and nonzero elements
    // of row i in the ELLPACK representation.
    int16_t  *ptr_A_col = &A_col[i * ell_width];
    bfloat16 *ptr_A_val = &A_val[i * ell_width];

    // Process the nonzero elements in groups of 32.
    for (uint32_t j = 0; j < ell_width; j += 32) {
        // Vector load: load matrix data into SIMD registers.
        vector_A_col = vector_load<32>(ptr_A_col);
        vector_A_val = vector_load<32>(ptr_A_val);

        // Scalar gather load: access x using noncontiguous column indices.
        aie::vector<bfloat16, 32> vector_x;
        for (int k = 0; k < 32; k++) {
            vector_x[k] = x[vector_A_col[k]];
        }

        // Vector multiply-accumulate.
        vector_acc = vector_mac(vector_acc, vector_A_val, vector_x);

        ptr_A_col += 32;
        ptr_A_val += 32;
    }

    // Reduce the partial sums and store the result in y[i].
    float row_sum = reduce_add(vector_acc);
    y[i] = (bfloat16)row_sum;
}