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
    cholmod_sdmult(A, 0, (double[2]){1,0}, (double[2]){0,0}, x, y, c);
    auto end = std::chrono::high_resolution_clock::now();

    double time_taken = std::chrono::duration<double>(end - start).count();

    cholmod_free_dense(&x, c);
    cholmod_free_dense(&y, c);
    return time_taken;
}

int main() {
    cholmod_common c;
    cholmod_start(&c);

    std::ofstream csv_file("benchmark_results.csv");
    csv_file << "Matrix_Size,Nonzeros,SpMV_Time(s)\n";

    std::vector<int> sizes = {1000, 5000, 10000, 20000};
    for (int n : sizes) {
        int nnz = n * 10;  // Set sparsity to 10x rows
        cholmod_sparse* A = generate_sparse_matrix(n, n, nnz, &c);

        if (!A) {
            std::cerr << "Matrix generation failed for size " << n << std::endl;
            continue;
        }

        double time_taken = benchmark_spmv(A, n, &c);
        std::cout << "SpMV for " << n << "x" << n << " matrix took " << time_taken << " seconds." << std::endl;

        csv_file << n << "," << nnz << "," << time_taken << "\n";

        cholmod_free_sparse(&A, &c);
    }

    csv_file.close();
    cholmod_finish(&c);
    return 0;
}