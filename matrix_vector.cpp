#include <chrono>
#include <iostream>
#include <cstdlib>
#include <omp.h>
#include "cholmod.h"

cholmod_sparse* generate_sparse_matrix(int n, size_t nnz, cholmod_common* c) {
    cholmod_triplet* T = cholmod_allocate_triplet(n, n, nnz, 0, CHOLMOD_REAL, c);
    if (!T) {
        std::cerr << "Failed to allocate triplet matrix" << std::endl;
        return nullptr;
    }
    
    double* values = static_cast<double*>(T->x);
    int* row_indices = static_cast<int*>(T->i);
    int* col_indices = static_cast<int*>(T->j);

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(nnz); i++) {
        row_indices[i] = rand() % n;
        col_indices[i] = rand() % n;
        values[i] = static_cast<double>(rand()) / RAND_MAX;
    }
    T->nnz = nnz;
    
    cholmod_sparse* A = cholmod_triplet_to_sparse(T, nnz, c);
    cholmod_free_triplet(&T, c);
    return A;
}

double benchmark_sparse_vector_multiplication(cholmod_sparse* A, cholmod_common* c) {
    int n = A->nrow;  
    cholmod_dense* x = cholmod_zeros(n, 1, CHOLMOD_REAL, c);
    if (!x) {
        std::cerr << "Failed to allocate dense vector x." << std::endl;
        return -1;
    }
    
    double* x_values = static_cast<double*>(x->x);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        x_values[i] = static_cast<double>(rand()) / RAND_MAX;
    }
    
    cholmod_dense* y = cholmod_zeros(n, 1, CHOLMOD_REAL, c);
    if (!y) {
        std::cerr << "Failed to allocate dense vector y." << std::endl;
        cholmod_free_dense(&x, c);
        return -1;
    }
    
    double alpha = 1.0, beta = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    cholmod_sdmult(A, 0, &alpha, &beta, x, y, c);
    auto end = std::chrono::high_resolution_clock::now();
    
    double time_taken = std::chrono::duration<double>(end - start).count();
    
    cholmod_free_dense(&x, c);
    cholmod_free_dense(&y, c);
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
    std::cout << "Using " << c.nthreads_max << " threads\n";
    
    size_t nnz = n * n / 100;
    std::cout << "Number of nonzeros: " << nnz << "\n";
    
    cholmod_sparse* A = generate_sparse_matrix(n, nnz, &c);
    if (!A) {
        std::cerr << "Matrix generation failed for size " << n << std::endl;
        cholmod_finish(&c);
        return 1;
    }
    
    double time_taken = benchmark_sparse_vector_multiplication(A, &c);
    if (time_taken >= 0) {
        std::cout << "Sparse matrix-vector multiplication for " << n << "x" << n
                  << " matrix took " << time_taken << " seconds." << std::endl;
    }
    
    cholmod_free_sparse(&A, &c);
    cholmod_finish(&c);
    
    return 0;
}
