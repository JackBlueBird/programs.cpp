#include <iostream>
#include <cmath>
#include <omp.h>
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
    g++ -o jacobi_omp.x laplace_jacobi_OMP.cpp -O3 -fopenmp
    g++ -o jacobi_omp.x laplace_jacobi_OMP.cpp -fopenmp -g // DUBUGGING
    Hot wo run:
    ./code.x 100 10 10
*/
/*
    code analyzer
    https://repo.hca.bsc.es/epic
    https://godbolt.org/
*/

int main (int argc, char* argv[]) {
    long int N, M; // number of solved cells and total cells per side
    long int i,j,k; // cartesian and linear cell indexes
    int iter, nIter; // iteration index and total number of iteration
    int displayStep;
    double *u_new, *u; // solution and solution of previous iteration
    double *u_temp; // additional pointer for swapping
    double *b; // storage for rhs
    double *res_vec; // residual history storage
    double res; // residual for current iteration
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
        displayStep = atoi(argv[3]); // third command line input is display step of the residual
    }
    M = N+2;
    h = (1.0)/(N+1);

    // Allocate memory
    u_new = (double *)malloc((N+2)*(N+2)*sizeof(double));    
    u = (double *)malloc((N+2)*(N+2)*sizeof(double));
    b = (double *)malloc((N+2)*(N+2)*sizeof(double));
    res_vec = (double *)malloc((N+2)*(N+2)*sizeof(double));

    // Set storages to 0
    for (k = 0; k < M*M; k++) {
        u_new[k] = 0.0;
        u[k] = 0.0;
        b[k] = 0.0;
        res_vec[k] = 0.0;
    }
    
    // Initial conditions and Right hand side
    for (i=1; i<M-1; i++) { // internal cells over i
        for (j=1; j<M-1; j++) { // internal cells over j
            k = j+M*i; // store using row major
            u_new[k] = value_init; // set initial value for u
            u[k] = value_init; // set initial value for u
            b[k] = h*h*value_force; // set internal cells right hand side
        }
    }
    
    auto start = std::chrono::high_resolution_clock::now();

    // Jacobi iterations cycle
    for (iter = 0; iter<nIter; iter++) {
        res = 0.0;
        #pragma omp parallel for collapse(2) reduction(+:res)
        for (i = 1; i < M-1; i++) {
            for (j = 1; j < M-1; j++) {
                k = j + M*i; // read and store using row major
                // u_new = (D-A)*u_prev + b
                // A = D-N (diagonal minus off diagonal part)
                u_new[k] = 0.25*(u[k+1]+u[k-1]+u[k+M]+u[k-M]+b[k]); // update u using stencil      
                // compute residual at previous iteration
                res += (4*u[k]-u[k+1]-u[k-1]-u[k+M]-u[k-M]-b[k])*(4*u[k]-u[k+1]-u[k-1]-u[k+M]-u[k-M]-b[k]);
            }
        }
        // store residual in history and print it
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