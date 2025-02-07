// benchmark_parallel.cpp
//
// This example benchmarks a sparse matrix Cholesky decomposition using SuiteSparse/CHOLMOD.
// It creates a symmetric positive–definite matrix in triplet form using a banded structure.
// The matrix is built in parallel with OpenMP. For each row i, we plan to fill entries
// for columns from max(0,i-b) up to i, ensuring that the diagonal entry is always included
// and set to (sum of absolute off–diagonals + 1.0) so that the row is strictly diagonally dominant.
//
// The user supplies two command–line arguments:
//    argv[1] = matrix dimension (n)
//    argv[2] = target number of nonzeros (nz_target)
// Note: If the “natural” total (i.e. if every row were completely filled) exceeds nz_target,
// then the last row is only partially filled (always including its diagonal).
//
// Compile (adjust SuiteSparse include/library paths as needed):
//   g++ -O2 -std=c++11 -fopenmp -I/path/to/suitesparse/include benchmark_parallel.cpp -o benchmark_parallel -lcholmod -lamd -lcolamd -lmetis
//
// Example (to target roughly 25e9 nonzeros):
//   ./benchmark_parallel 100000000 25000000000
//
// WARNING: Running with such huge parameters will allocate hundreds of GB of memory.

#include <cholmod.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <omp.h>

int main(int argc, char **argv)
{
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <matrix_dimension> <target_num_nonzeros>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // --- Parse command-line arguments ---
    // Use a 64-bit type (SuiteSparse_long) for huge indices.
    SuiteSparse_long n = atoll(argv[1]);       // matrix dimension (n x n)
    SuiteSparse_long nz_target = atoll(argv[2]); // target total nonzeros

    // Compute an approximate bandwidth.
    // (For a lower–triangular band, each row i would normally have count = min(b+1, i+1) entries.)
    double avg_per_row = static_cast<double>(nz_target) / static_cast<double>(n);
    // Here we choose: b = (avg_per_row > 1 ? avg_per_row - 1 : 0)
    SuiteSparse_long b = (avg_per_row > 1) ? static_cast<SuiteSparse_long>(avg_per_row) - 1 : 0;

    std::printf("Matrix dimension: %lld x %lld\n", n, n);
    std::printf("Target nonzeros: %lld, approximate bandwidth: %lld (off-diagonals per full row ~%lld)\n",
                nz_target, b, b);

    // --- Determine per–row counts ---
    // For each row i, if fully filled we would have:
    //    full_count[i] = min(b+1, i+1)
    // We accumulate until we reach (or slightly exceed) nz_target.
    std::vector<SuiteSparse_long> row_counts;
    row_counts.reserve(n);
    SuiteSparse_long total_nnz = 0;
    SuiteSparse_long final_rows = 0;
    for (SuiteSparse_long i = 0; i < n; i++) {
        SuiteSparse_long full_count = (i < b ? i + 1 : b + 1);
        if (total_nnz + full_count > nz_target) {
            // In the final row, we fill only a partial set of entries.
            SuiteSparse_long remaining = nz_target - total_nnz;
            // Always reserve at least one slot for the diagonal.
            if (remaining < 1) remaining = 1;
            row_counts.push_back(remaining);
            total_nnz += remaining;
            final_rows = i + 1;
            break;
        } else {
            row_counts.push_back(full_count);
            total_nnz += full_count;
        }
    }
    // If nz_target exceeds the natural total, use the full matrix.
    if (total_nnz < nz_target) {
        final_rows = n;
        std::printf("Warning: Full matrix natural nnz (%lld) is less than target (%lld). Using full matrix.\n",
                    total_nnz, nz_target);
    }
    std::printf("Rows to fill: %lld, Total nonzeros: %lld\n", final_rows, total_nnz);

    // --- Compute row offsets (prefix sum) ---
    // offsets[i] is the starting index in the triplet arrays for row i.
    std::vector<SuiteSparse_long> offsets(row_counts.size() + 1, 0);
    for (size_t i = 0; i < row_counts.size(); i++) {
        offsets[i+1] = offsets[i] + row_counts[i];
    }
    if (offsets.back() != total_nnz) {
        std::fprintf(stderr, "Error computing row offsets.\n");
        return EXIT_FAILURE;
    }

    // --- Initialize CHOLMOD ---
    cholmod_common c;
    cholmod_start(&c);

    // Allocate a cholmod_triplet structure.
    cholmod_triplet *T = cholmod_allocate_triplet(n, n, total_nnz, 0, CHOLMOD_REAL, &c);
    if (!T) {
        std::fprintf(stderr, "Error: Unable to allocate cholmod_triplet.\n");
        cholmod_finish(&c);
        return EXIT_FAILURE;
    }
    // Tell CHOLMOD that the matrix is symmetric and only the lower triangle is stored.
    T->stype = -1;
    T->nnz = total_nnz; // Set the number of nonzeros

    // For easier access, get pointers to the arrays.
    SuiteSparse_long *Ti = static_cast<SuiteSparse_long*>(T->i);
    SuiteSparse_long *Tj = static_cast<SuiteSparse_long*>(T->j);
    double *Tx = static_cast<double*>(T->x);

    // --- Fill the triplet in parallel ---
    // For each row i in [0, final_rows), we fill row_counts[i] entries.
    // We always ensure that the diagonal entry is produced and set to (sum_offdiag + 1.0).
    // The off-diagonal entries (columns j in [max(0, i-b), i)) are given small random values.
    #pragma omp parallel for schedule(dynamic)
    for (SuiteSparse_long i = 0; i < final_rows; i++) {
        SuiteSparse_long offset = offsets[i];    // starting index for row i
        SuiteSparse_long count = row_counts[i];    // number of entries to fill in row i

        // For row i, the full (if not truncated) set would cover columns j from:
        //    j_min = max(0, i - b)  to  (i-1)  for off-diagonals, and then column i (diagonal).
        SuiteSparse_long j_min = (i > b ? i - b : 0);
        SuiteSparse_long possible_offdiag = i - j_min;  // maximum off-diagonals possible

        // We must reserve one slot for the diagonal.
        SuiteSparse_long offdiag_count = (count - 1 < possible_offdiag ? count - 1 : possible_offdiag);

        double sum_abs = 0.0;

        // Use a thread–local seed (based on row index) for random numbers.
        unsigned int seed = static_cast<unsigned int>(i + 12345);

        // Fill off–diagonal entries.
        for (SuiteSparse_long k = 0; k < offdiag_count; k++) {
            SuiteSparse_long j_val = j_min + k;
            Ti[offset + k] = i;
            Tj[offset + k] = j_val;
            // Generate a random value in [-0.05, 0.05]
            double r = (static_cast<double>(rand_r(&seed)) / RAND_MAX) - 0.5;
            double value = r * 0.1;
            Tx[offset + k] = value;
            sum_abs += std::fabs(value);
        }
        // Fill the diagonal entry in the next slot.
        SuiteSparse_long diag_index = offset + offdiag_count;
        Ti[diag_index] = i;
        Tj[diag_index] = i;
        Tx[diag_index] = sum_abs + 1.0; // ensures that |diag| > sum of off-diagonals

        // (In a full row, count should equal possible_offdiag + 1.
        //  In a truncated final row, we still always include the diagonal.)
    }

    // --- Convert triplet to compressed–column form ---
    cholmod_sparse *A = cholmod_triplet_to_sparse(T, total_nnz, &c);
    if (!A) {
        std::fprintf(stderr, "Error: Unable to convert triplet to sparse matrix.\n");
        cholmod_free_triplet(&T, &c);
        cholmod_finish(&c);
        return EXIT_FAILURE;
    }
    // The triplet is no longer needed.
    cholmod_free_triplet(&T, &c);

    // --- Benchmark: Cholesky Decomposition ---
    std::printf("Starting Cholesky decomposition...\n");
    double t_start = omp_get_wtime();

    // Analyze the sparsity pattern.
    cholmod_factor *L = cholmod_analyze(A, &c);
    if (!L) {
        std::fprintf(stderr, "Error during cholmod_analyze.\n");
        cholmod_free_sparse(&A, &c);
        cholmod_finish(&c);
        return EXIT_FAILURE;
    }
    // Perform the numerical factorization.
    int status = cholmod_factorize(A, L, &c);
    if (!status) {
        std::fprintf(stderr, "Error during cholmod_factorize.\n");
    }
    double t_end = omp_get_wtime();
    std::printf("Cholesky decomposition completed in %.2f seconds.\n", t_end - t_start);

    // --- Cleanup ---
    cholmod_free_factor(&L, &c);
    cholmod_free_sparse(&A, &c);
    cholmod_finish(&c);

    return EXIT_SUCCESS;
}

