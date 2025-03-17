#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>
#include "mkl.h"

using namespace std;

void generate_random_sparse_csr(int n, int nnz_per_row,
                                vector<int>& row_ptr, vector<int>& col_idx, vector<double>& values) {
  int nnz = n * nnz_per_row;
  row_ptr.resize(n + 1, 0);
  col_idx.reserve(nnz);
  values.reserve(nnz);

  #pragma omp parallel
  {
    unsigned int seed = omp_get_thread_num() + time(NULL);
    #pragma omp for
    for (int i = 0; i < n; i++) {
      row_ptr[i + 1] = row_ptr[i] + nnz_per_row;
      for (int j = 0; j < nnz_per_row; j++) {
        int col = rand_r(&seed) % n;
        double val = static_cast<double>(rand_r(&seed)) / RAND_MAX;
        #pragma omp critical
        {
            col_idx.push_back(col);
            values.push_back(val);
        }
      }
    }
  }
}

double benchmark_sparse_multiply(MKL_INT n,
                                 sparse_matrix_t A, sparse_matrix_t B) {
  sparse_matrix_t C;
  struct matrix_descr descr;
  descr.type = SPARSE_MATRIX_TYPE_GENERAL;

  auto start = chrono::high_resolution_clock::now();
  sparse_status_t status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, A, B, &C);
  auto end = chrono::high_resolution_clock::now();

  if (status != SPARSE_STATUS_SUCCESS) {
    cerr << "MKL SpMM failed\n";
    return -1;
  }

  mkl_sparse_destroy(C);
  return chrono::duration<double>(end - start).count();
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    cerr << "Usage: " << argv[0] << " <matrix_size> <num_threads>\n";
    return 1;
  }

  int n = atoi(argv[1]);
  int num_threads = atoi(argv[2]);
  omp_set_num_threads(num_threads);

  cout << "using " << num_threads << " threads\n";

  int nnz_per_row = n / 100;
  if (nnz_per_row < 1){
    nnz_per_row = 1;
  } 

  cout << "generating sparse matrix A\n";

  vector<int> row_ptr_A, col_idx_A;
  vector<double> values_A;
  generate_random_sparse_csr(n, nnz_per_row, row_ptr_A, col_idx_A, values_A);

  cout << "generating sparse matrix B\n";
  vector<int> row_ptr_B, col_idx_B;
  vector<double> values_B;
  generate_random_sparse_csr(n, nnz_per_row, row_ptr_B, col_idx_B, values_B);

  sparse_matrix_t A, B;
  cout << "converting A to sparse format\n";
  mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, n, n,
                          row_ptr_A.data(), row_ptr_A.data() + 1,
                          col_idx_A.data(), values_A.data());

  cout << "converting B to sparse format\n";
  mkl_sparse_d_create_csr(&B, SPARSE_INDEX_BASE_ZERO, n, n,
                          row_ptr_B.data(), row_ptr_B.data() + 1,
                          col_idx_B.data(), values_B.data());

  cout << "benchmarking MKL sparse matrix-matrix multiplication...\n";

  double elapsed = benchmark_sparse_multiply(n, A, B);
  if (elapsed >= 0) {
    cout << "MKL SpMM for " << n << "x" << n << " took " << elapsed << " seconds\n";
  }

  mkl_sparse_destroy(A);
  mkl_sparse_destroy(B);
  return 0;
}
