#include <iostream>
#include <cmath>
#include <chrono>
/*
    Author: Giacomo Zuccarino.
    Date: February 2025.
*/
/*
    This program solves the 2D laplace equation 
    on the square [0,1] x [0,1]
    using jacobi iteration as linear system solver
*/
/*
    How to compile on linux:
    g++ -o jacobi.x laplace_jacobi.cpp -O3
    g++ -o code.x laplace_jacobi.cpp -g // DUBUGGING
    g++ -O2 -ftree-vectorize -ftree-vectorizer-verbose=4 -fopt-info-vec laplace_jacobi.cpp -o jacobi_test.x // CHECK VECTORIZATION
    // works with both -O2 or -O3
    Hot wo run:
    ./jacobi.x 1000 10 1
*/
/*
    code analyzer
    https://repo.hca.bsc.es/epic
    https://godbolt.org/
*/

int main (int argc, char* argv[]) {
    long int N, M; // number of solved cells and total cells per side
    long int i,j,k; // cartesian and linear cell indexes
    int iter, nIter; // iteration index and total number
    int displayStep;
    double *u, *u_new; // solution and solution of previous iteration
    double *u_temp; // additional pointer for swapping
    double *b; // storage for rhs
    double *res_vec;  // residual vector
    double res;
    double value_init = 0.5;
    double value_force = 1.0;
    double h; // discretization step

    N = 4;
    nIter = 1;
    displayStep = 1;
    if (argc >= 2) { // first command line input is linear size of the problem
        N = atoi(argv[1]);
    }
    if (argc >= 3) { // second command line input is number of iterations
        nIter = atoi(argv[2]);
    }
    if (argc >= 4) {
        displayStep = atoi(argv[3]); // this command line input is display step of the residual
    }
    M = N+2;
    h = (1.0)/(N+1);

    // Allocate memory
    u = (double *)malloc((N+2)*(N+2)*sizeof(double));    
    u_new = (double *)malloc((N+2)*(N+2)*sizeof(double));
    b = (double *)malloc((N+2)*(N+2)*sizeof(double));
    res_vec = (double *)malloc(nIter*sizeof(double));

    // Set storages to 0
    for (k = 0; k < M*M; k++) {
        u[k] = 0.0;
        u_new[k] = 0.0;
        b[k] = 0.0;
    }
    
    // Initial conditions and Right hand side
    for (i=1; i<M-1; i++) { // internal cells over i
        for (j=1; j<M-1; j++) { // internal cells over j
            k = j+M*i; // row major
            u[k] = value_init;
            u_new[k] = value_init;
            b[k] = h*h*value_force;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    // Jacobi iteration
    for (iter = 0; iter<nIter; iter++) {
        res = 0.0;
        for (i = 1; i < M-1; i++) {
            for (j = 1; j < M-1; j++) {
                k = j + M*i; // row major
                u_new[k] = 0.25*(u[k+1]+u[k-1]+u[k+M]+u[k-M]+b[k]);
                // r = A*u -b
                // residual is actually from previous iteration to spare computational cost
                res += (4*u[k]-u[k+1]-u[k-1]-u[k+M]-u[k-M]-b[k])*(4*u[k]-u[k+1]-u[k-1]-u[k+M]-u[k-M]-b[k]);          
            }
        }

        res_vec[iter] = h*sqrt(res);
        if (iter%displayStep == 0) {
            std::cout << "iter: " << iter <<  " , res: " << res_vec[iter] << std::endl;
        }
        
        //swap pointers
        u_temp = u;
        u = u_new;
        u_new = u_temp;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Function execution time: " << duration.count() << " microseconds" << std::endl;
    std::cout << "prevent compiler optimization: "<< res_vec[0] << std::endl;

    free(u_new);
    free(u);
    free(b);
    free(res_vec);
}