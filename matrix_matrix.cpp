#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>
#include "mkl.h"

using namespace std;

void generate_random_sparse_csr(long long n, long long nnz_per_row,
                                vector<long long>& row_ptr, vector<long long>& col_idx, vector<double>& values) {
  long long nnz = (long long)n * (long long)nnz_per_row;
  row_ptr.resize(n + 1, 0);
  col_idx.resize(nnz);
  values.resize(nnz);
  
  for (int i = 0; i <= n; i++) {
    row_ptr[i] = i * nnz_per_row;
  }

  #pragma omp parallel
  {
    unsigned int seed = omp_get_thread_num() + time(NULL);
    #pragma omp for
    for (int i = 0; i < n; i++) {
      long long row_offset = i * nnz_per_row;
      for (int j = 0; j < nnz_per_row; j++) {
        long long idx = row_offset + j;
        int col = rand_r(&seed) % n;
        double val = static_cast<double>(rand_r(&seed)) / RAND_MAX;
        col_idx[idx] = col;
        values[idx] = val;
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
    cerr << "MKL SpMM failed, status code: " << status << "\n";
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
  cout << "size of mkl int: " << sizeof(MKL_INT) << endl;
  long long n = atoi(argv[1]);
  int num_threads = atoi(argv[2]);
  omp_set_num_threads(num_threads);
  mkl_set_num_threads(num_threads);


  cout << "using " << num_threads << " threads" << endl;

  long long nnz_per_row = n / 100;
  if (nnz_per_row < 1){
    nnz_per_row = 1;
  } 

  cout << "generating sparse matrix A" << endl;
  auto start = chrono::high_resolution_clock::now();
  vector<long long> row_ptr_A, col_idx_A;
  vector<double> values_A;
  generate_random_sparse_csr(n, nnz_per_row, row_ptr_A, col_idx_A, values_A);

  cout << "generating sparse matrix B" << endl;
  vector<long long> row_ptr_B, col_idx_B;
  vector<double> values_B;
  generate_random_sparse_csr(n, nnz_per_row, row_ptr_B, col_idx_B, values_B);
  
  auto end = chrono::high_resolution_clock::now();
  double time_taken = chrono::duration<double>(end - start).count();
  cout << "time taken for generating random sparse A and B: " << time_taken << " seconds" << endl;
  sparse_matrix_t A, B;
  cout << "converting A to sparse format" << endl;
  start = chrono::high_resolution_clock::now();
  mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, n, n,
                          row_ptr_A.data(), row_ptr_A.data() + 1,
                          col_idx_A.data(), values_A.data());

  cout << "converting B to sparse format" << endl;
  mkl_sparse_d_create_csr(&B, SPARSE_INDEX_BASE_ZERO, n, n,
                          row_ptr_B.data(), row_ptr_B.data() + 1,
                          col_idx_B.data(), values_B.data());
  
  end = chrono::high_resolution_clock::now();
  time_taken = chrono::duration<double>(end - start).count();
  cout << "time taken for creating csr: " << time_taken << " seconds" << endl;
  cout << "benchmarking MKL sparse matrix-matrix multiplication..." << endl;

  double elapsed = benchmark_sparse_multiply(n, A, B);
  if (elapsed >= 0) {
    cout << "MKL SpMM for " << n << "x" << n << " took " << elapsed << " seconds" << endl;
  }

  mkl_sparse_destroy(A);
  mkl_sparse_destroy(B);
  return 0;
}
