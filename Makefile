main:
	g++ -o main main.cpp -std=c++17 -fopenmp -lcholmod -lamd -lcolamd -lccolamd -lcamd -lsuitesparseconfig -lm
vector:
	g++ -o matrix_vector matrix_vector.cpp -std=c++17 -fopenmp -lcholmod -lamd -lcolamd -lccolamd -lcamd -lsuitesparseconfig -lm
