vector:
	g++ -o matrix_vector matrix_vector.cpp -std=c++17 -fopenmp -lcholmod -lamd -lcolamd -lccolamd -lcamd -lsuitesparseconfig -lm
matrix:
	g++ -o matrix matrix_matrix.cpp -std=c++17 -fopenmp -DMKL_ILP64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
test:
	g++ -o open test_open.cpp -std=c++17 -fopenmp -lcholmod -lamd -lcolamd -lccolamd -lcamd -lsuitesparseconfig -lm
