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
    for (int i = 0; i < nnz; i++) {
        row_indices[i] = rand() % n;
        col_indices[i] = rand() % n;
        values[i] = static_cast<double>(rand()) / RAND_MAX;
    }
    T->nnz = nnz;
    cholmod_sparse* A = cholmod_triplet_to_sparse(T, nnz, c);
    cholmod_free_triplet(&T, c);
    return A;
}

double benchmark_sparse_multiplication(cholmod_sparse* A, cholmod_sparse* B, cholmod_common* c) {
    auto start = std::chrono::high_resolution_clock::now();
    cholmod_sparse* C = cholmod_ssmult(A, B, 0, 1, 1, c); 
    auto end = std::chrono::high_resolution_clock::now();

    double time_taken = std::chrono::duration<double>(end - start).count();

    if (C) {
        cholmod_free_sparse(&C, c);
    } else {
        std::cerr << "Matrix multiplication failed!" << std::endl;
        return -1;
    }

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

    // Generate two random sparse matrices A and B
    cholmod_sparse* A = generate_sparse_matrix(n, nnz, &c);
    cholmod_sparse* B = generate_sparse_matrix(n, nnz, &c);

    if (!A || !B) {
        std::cerr << "Matrix generation failed for size " << n << std::endl;
        cholmod_finish(&c);
        return 1;
    }

    // Perform and benchmark sparse matrix multiplication
    double time_taken = benchmark_sparse_multiplication(A, B, &c);
    if (time_taken >= 0) {
        std::cout << "Sparse matrix multiplication for " << n << "x" << n
                  << " matrix took " << time_taken << " seconds." << std::endl;
    }

    // Clean up
    cholmod_free_sparse(&A, &c);
    cholmod_free_sparse(&B, &c);
    cholmod_finish(&c);

    return 0;
}
