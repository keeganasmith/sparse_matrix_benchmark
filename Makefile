vector:
	g++ -o matrix_vector matrix_vector.cpp -std=c++17 -fopenmp -lcholmod -lamd -lcolamd -lccolamd -lcamd -lsuitesparseconfig -lm
