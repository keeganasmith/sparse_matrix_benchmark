#include <chrono>
#include <iostream>
#include <cstdlib>
#include <omp.h>
#include "cholmod.h"
using std::string, std::to_string, std::cout;
long long generate_random_64bit(unsigned int& myseed){
  long long num_one = static_cast<long long>(time(NULL) + rand_r(&myseed));
  long long num_two = static_cast<long long>(time(NULL) + rand_r(&myseed));
  return (num_one << 32) | num_two;
}
cholmod_sparse* generate_sparse_matrix(long long n, long long nnz, cholmod_common* c,  std::chrono::time_point<std::chrono::high_resolution_clock> start_time, string matrix_name) {
  cholmod_triplet* T = cholmod_l_allocate_triplet(n, n, nnz, 0, CHOLMOD_REAL, c);
  if (!T) {
      std::cerr << "Failed to allocate triplet matrix" << std::endl;
      return nullptr;
  }
  auto curr_time = std::chrono::high_resolution_clock::now();
  cout << "generating sparse matrix values at t = " << std::chrono::duration<double>(curr_time - start_time).count() << "\n"; 
  double* values = static_cast<double*>(T->x);
  long long* row_indices = static_cast<long long*>(T->i);
  long long* col_indices = static_cast<long long*>(T->j);

  #pragma omp parallel
  {
    unsigned int myseed = omp_get_thread_num();
    #pragma omp for
    for (long long i = 0; i < (long long) nnz; i++) {
        row_indices[i] = (time(NULL) + rand_r(&myseed)) % n;
        col_indices[i] = (time(NULL) + rand_r(&myseed)) % n;
        values[i] = double(time(NULL) + rand_r(&myseed)) / double(RAND_MAX);
        
    }
  }
  T->nnz = nnz;
  curr_time = std::chrono::high_resolution_clock::now();
  cout << "converting triplet to sparse matrix at t = " << std::chrono::duration<double>(curr_time - start_time).count() << "\n";
  cholmod_sparse* A = cholmod_l_triplet_to_sparse(T, nnz, c);
  cholmod_l_free_triplet(&T, c);
  string file_name = matrix_name + ".mtx";
  FILE *f = fopen(file_name.c_str(), "w");
  cholmod_l_write_sparse(f, A, nullptr, nullptr, c);
  return A;
}

cholmod_dense* generate_random_dense_vector(cholmod_common* c, long long n, bool is_x_vector){
  cholmod_dense* x = cholmod_l_zeros(n, 1, CHOLMOD_REAL, c);
  if (!x) {
      std::cerr << "Failed to allocate dense vector x." << std::endl;
      return nullptr;
  }
  double* x_values = static_cast<double*>(x->x);
  #pragma omp parallel
  {
    unsigned int myseed = omp_get_thread_num();
    #pragma omp parallel for
    for (long long i = 0; i < n; i++) {
      x_values[i] = static_cast<double>(time(NULL) + rand_r(&myseed));
    }
  }
  cholmod_l_print_dense(x, "Dense Vector", c);
  return x;
}



cholmod_dense* generate_B(cholmod_common* c, cholmod_sparse* A, long long n){
  cholmod_dense* x = generate_random_dense_vector(c, n, false);
  cholmod_dense* B = cholmod_l_zeros(A->nrow, 1, CHOLMOD_REAL, c);
  double alpha = 1.0;
  double beta = 0.0;
  cholmod_l_sdmult(A, 0, &alpha, &beta, x, B, c);
  cholmod_l_free_dense(&x, c);
  return B;
}

double benchmark_sparse_vector_multiplication(cholmod_sparse* A, cholmod_common* c,  std::chrono::time_point<std::chrono::high_resolution_clock>& start_time) {
  long long n = A->nrow;  
  auto curr_time = std::chrono::high_resolution_clock::now();
  cout << "generating dense vectors at t = " << std::chrono::duration<double>(curr_time - start_time).count() << "\n";
  cholmod_dense* x = generate_random_dense_vector(c, n, true);
  cholmod_dense* y = generate_random_dense_vector(c, n, false); 
  double alpha = 1.0, beta = 0.0;
  auto start = std::chrono::high_resolution_clock::now(); 
  cout << "starting multiplication at t = " << std::chrono::duration<double>(start - start_time).count() << "\n";
  cholmod_l_sdmult(A, 0, &alpha, &beta, x, y, c);
  auto end = std::chrono::high_resolution_clock::now();
  
  double time_taken = std::chrono::duration<double>(end - start).count();
  
  cholmod_l_free_dense(&x, c);
  cholmod_l_free_dense(&y, c);
  return time_taken;
}
double benchmark_sparse_solve(cholmod_sparse* A, cholmod_dense* B, cholmod_common* c){
  cout << "made it to factorization\n";
  cholmod_factor* L = cholmod_l_analyze(A, c);
  if (!L) {
      std::cerr << "cholmod_analyze failed.\n";
      return -1;
  } 
  if (!cholmod_factorize(A, L, c)) {
    std::cerr << "cholmod_factorize failed.\n";
    cholmod_l_free_factor(&L, c);
    return -1;
  }
  cout << "Finished analyzing, starting solving\n";
  auto start = std::chrono::high_resolution_clock::now();
  cholmod_sparse* X = (cholmod_sparse*)cholmod_l_solve(CHOLMOD_A, L, B, c);
  auto end = std::chrono::high_resolution_clock::now();
  cholmod_l_free_sparse(&X, c);
  cholmod_l_free_factor(&L, c);
  cout << "solve finished\n";
  std::chrono::duration<double> elapsed = end - start;
  return elapsed.count();
}
int main(int argc, char* argv[]) {
  auto start_time = std::chrono::high_resolution_clock::now(); 
  if (argc != 3) {
      std::cerr << "Usage: " << argv[0] << " <matrix_size> <thread_num>" << std::endl;
      return 1;
  }
  
  long long n = std::atoll(argv[1]);
  if (n <= 0) {
      std::cerr << "Error: Matrix size must be a positive integer." << std::endl;
      return 1;
  }
  int num_threads = std::atoi(argv[2]);
  omp_set_num_threads(num_threads);
  cholmod_common c;
  c.chunk = 1;
  cholmod_l_start(&c);
  std::cout << "Using " << c.nthreads_max << " threads\n";
  
  size_t nnz = n * n / 100;
  std::cout << "Number of nonzeros: " << nnz << "\n";
  cout << "Reading A from sparse_matrix.sp...\n";
  FILE *f = fopen("matrix_A.mtx", "r");
  int type;
  cholmod_sparse* A = (cholmod_sparse*) cholmod_l_read_matrix(f, 1, &type, &c);
  if(A == nullptr || type != CHOLMOD_SPARSE){
    cout << "Type was not cholmod sparse or A was nullptr, generating A...\n";
    A = generate_sparse_matrix(n, nnz, &c, start_time, "matrix_A");
  }
  if (!A) {
    std::cerr << "Matrix generation failed for size " << n << std::endl;
    cholmod_l_finish(&c);
    return 1;
  }

  cout << "read A, generating B now\n";
  cholmod_dense* B = generate_B(&c, A, n);
  if(!B){
    std::cerr << "vector generation failed for vector B\n";
    cholmod_l_finish(&c);
    return 1;
  }
  cholmod_l_print_sparse(A, "Sparse Matrix", &c);

  //double time_taken = benchmark_sparse_vector_multiplication(A, &c, start_time);
  double time_taken = benchmark_sparse_solve(A, B, &c); 
  if (time_taken >= 0) {
      std::cout << "Sparse matrix-vector multiplication for " << n << "x" << n
                << " matrix took " << time_taken << " seconds." << std::endl;
  }
  
  cholmod_l_free_sparse(&A, &c);
  cholmod_l_free_dense(&B, &c);
  cholmod_l_finish(&c);
  
  return 0;
}
