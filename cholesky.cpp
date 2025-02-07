#include <chrono>
#include <iostream>
#include <cstdlib>
#include "cholmod.h"
#include <omp.h>
cholmod_sparse* generate_symmetric_sparse_matrix(int n, size_t nnz, cholmod_common* c) {
    cholmod_triplet* T = cholmod_allocate_triplet(n, n, nnz + n, 1, CHOLMOD_REAL, c);
    if (!T) {
        std::cerr << "Failed to allocate triplet matrix" << std::endl;
        return nullptr;
    }

    double* values = static_cast<double*>(T->x);
    int* row_indices = static_cast<int*>(T->i);
    int* col_indices = static_cast<int*>(T->j);

    #pragma omp parallel for
    for (int i = 0; i < nnz; i++) {
        int r = rand() % n;
        int c = rand() % n;
        if (r > c) std::swap(r, c);

        row_indices[i] = r;
        col_indices[i] = c;
        values[i] = static_cast<double>(rand()) / RAND_MAX + 1.0;
    }

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        row_indices[nnz + i] = i;
        col_indices[nnz + i] = i;
        values[nnz + i] = n; 
    }

    T->nnz = nnz + n;

    cholmod_sparse* A = cholmod_triplet_to_sparse(T, nnz + n, c);
    cholmod_free_triplet(&T, c);
    return A;
}
double benchmark_cholesky(cholmod_sparse* A, cholmod_common* c) {
    cholmod_factor* L = cholmod_analyze_p(A, nullptr, nullptr, 1, c);;
    if (cholmod_check_sparse(A, c) != 1) {
      std::cerr << "Matrix A is not valid." << std::endl;
    }
    if (c->status == CHOLMOD_OUT_OF_MEMORY) {
      std::cerr << "CHOLMOD ran out of memory!" << std::endl;
    }
    cholmod_print_sparse(A, "Matrix A", c);
    if (cholmod_check_sparse(A, c) != 1) {
        std::cerr << "Matrix A failed CHOLMOD validation." << std::endl;
    }
    std::cout << c->status << "\n";
    if (!L) {
        std::cerr << "Factorization analysis failed." << std::endl;
        return -1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    cholmod_factorize(A, L, c);
    auto end = std::chrono::high_resolution_clock::now();

    double time_taken = std::chrono::duration<double>(end - start).count();

    cholmod_free_factor(&L, c);
    return time_taken;
}

int main(int argc, char* argv[]) {
    omp_set_num_threads(96);
    
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return 1;
    }

    size_t n = std::atoi(argv[1]);
    if (n <= 0) {
        std::cerr << "Error: Matrix size must be a positive integer." << std::endl;
        return 1;
    }

    cholmod_common c;
    cholmod_start(&c);
    
    std::cout << "using " << c.nthreads_max << " threads\n";

    size_t nnz = n * n / 100;
    std::cout << "Num nonzeroes: " << nnz << "\n";
    cholmod_sparse* A = generate_symmetric_sparse_matrix(n, nnz, &c);

    if (!A) {
        std::cerr << "Matrix generation failed for size " << n << std::endl;
        cholmod_finish(&c);
        return 1;
    }

    double time_taken = benchmark_cholesky(A, &c);
    std::cout << "Cholesky factorization for " << n << "x" << n << " matrix took " << time_taken << " seconds." << std::endl;

    cholmod_free_sparse(&A, &c);
    cholmod_finish(&c);

    return 0;
}
