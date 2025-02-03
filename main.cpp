#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include "cholmod.h"

cholmod_sparse* generate_sparse_matrix(int rows, int cols, int nnz, cholmod_common* c) {
    cholmod_triplet* T = cholmod_allocate_triplet(rows, cols, nnz, 0, CHOLMOD_REAL, c);
    if (!T) {
        std::cerr << "Failed to allocate triplet matrix" << std::endl;
        return nullptr;
    }
    
    double* values = static_cast<double*>(T->x);
    int* row_indices = static_cast<int*>(T->i);
    int* col_indices = static_cast<int*>(T->j);

    for (int i = 0; i < nnz; i++) {
        row_indices[i] = rand() % rows;
        col_indices[i] = rand() % cols;
        values[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    T->nnz = nnz;
    cholmod_sparse* A = cholmod_triplet_to_sparse(T, nnz, c);
    cholmod_free_triplet(&T, c);
    return A;
}

double benchmark_spmv(cholmod_sparse* A, int n, cholmod_common* c) {
    cholmod_dense* x = cholmod_ones(n, 1, CHOLMOD_REAL, c);
    cholmod_dense* y = cholmod_zeros(n, 1, CHOLMOD_REAL, c);

    auto start = std::chrono::high_resolution_clock::now();
    double a[2] = {1, 0};
    double b[2] = {0, 0}; 
    cholmod_sdmult(A, 0, a, b, x, y, c);
    auto end = std::chrono::high_resolution_clock::now();

    double time_taken = std::chrono::duration<double>(end - start).count();

    cholmod_free_dense(&x, c);
    cholmod_free_dense(&y, c);
    return time_taken;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return 1;
    }

    int n = std::atoi(argv[1]);  // Convert argument to integer
    if (n <= 0) {
        std::cerr << "Error: Matrix size must be a positive integer." << std::endl;
        return 1;
    }

    cholmod_common c;
    cholmod_start(&c);
    std::string csv_file_name = "benchmark_results_" + std::to_string(n) + ".csv";
    std::ofstream csv_file(csv_file_name);
    csv_file << "Matrix_Size,Nonzeros,SpMV_Time(s)\n";

    int nnz = n * 10;  // Set sparsity to 10x rows
    cholmod_sparse* A = generate_sparse_matrix(n, n, nnz, &c);

    if (!A) {
        std::cerr << "Matrix generation failed for size " << n << std::endl;
        cholmod_finish(&c);
        return 1;
    }

    double time_taken = benchmark_spmv(A, n, &c);
    std::cout << "SpMV for " << n << "x" << n << " matrix took " << time_taken << " seconds." << std::endl;

    csv_file << n << "," << nnz << "," << time_taken << "\n";

    // Free memory
    cholmod_free_sparse(&A, &c);
    csv_file.close();
    cholmod_finish(&c);

    return 0;
}