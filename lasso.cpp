#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <tuple>
#define EIGEN_USE_MKL_ALL
#include <eigen3/Eigen/Dense>
#include <mkl.h>

#include "lasso.h"


//using namespace std; 
//using namespace Eigen; 

tuple<VectorXf,double>
lasso (MatrixXf A,  VectorXf b, float lambda, int MAX_ITER, float RELTOL, float ABSTOL, float rho, MPI_Comm comm)
 {
        /*const int MAX_ITER  = 50;
        const float RELTOL = 1e-2;
        const float ABSTOL = 1e-4;*/
        /*
         * Some bookkeeping variables for MPI. The 'rank' of a process is its numeric id
         * in the process pool. For example, if we run a program via `mpirun -np 4 foo', then
         * the process ranks are 0 through 3. Here, N and size are the total number of processes 
         * running (in this example, 4).
         */

        int rank;
        int size;

        MPI_Comm_rank(comm, &rank); // Determine current running process
        MPI_Comm_size(comm, &size); // Total number of processes
        float N = (float) size;             // Number of subsystems/slaves for ADMM

        /* Read in local data */

        //int skinny;           // A flag indicating whether the matrix A is fat or skinny


	/*
	 * For simple data where m is not so greater than n, then the matrix is copied
	 * as such and estimates are calculated. This will not gain anything from mpi_lasso.
	 * But for very large dataset the input matrix is chunked and individual estimates 
	 * are calculated by the each core and communicating through Allreduce, thereby
	 * distributing the problem across admm_cores. This strategy will greatly boost the computation time. 
	 */

	int m = A.rows();
	int n = A.cols();
      
	const int skinny = (m >= n);

        /*
         * These are all variables related to ADMM itself. There are many
         * more variables than in the Matlab implementation because we also
         * require vectors and matrices to store various intermediate results.
         * The naming scheme follows the Matlab version of this solver.
         */

        //float rho = 1.0;

         VectorXf   x   =  VectorXf::Zero(n);
         VectorXf   u   =  VectorXf::Zero(n);
         VectorXf   z   =  VectorXf::Zero(n);
         VectorXf   y   =  VectorXf::Zero(n);
         VectorXf   r   =  VectorXf::Zero(n);
      	 VectorXf   zprev =  VectorXf::Zero(n);
         VectorXf   zdiff =  VectorXf::Zero(n);
         VectorXf   q   =  VectorXf::Zero(n);
         VectorXf   w   =  VectorXf::Zero(n);
         VectorXf   Aq  =  VectorXf::Zero(m);
         VectorXf   p   =  VectorXf::Zero(m);
         VectorXf   Atb         =  VectorXf::Zero(n);

        float send[3]; // an array used to aggregate 3 scalars at once
        float recv[3]; // used to receive the results of these aggregations

        float nxstack  = 0;
        float nystack  = 0;
        float prires   = 0;
        float dualres  = 0;
        float eps_pri  = 0;
        float eps_dual = 0;

	/*sgemm variables */
	/*char *transA = "N";
        char *transB = "T";
        float alpha  = 1.0;
        float beta = 0.0;*/ 
 

        Atb = A.transpose()*b; // Atb = A^T b

         /*
         * The lasso regularization parameter here is just hardcoded
         * to 0.5 for simplicity. Using the lambda_max heuristic would require
         * network communication, since it requires looking at the *global* A^T b.
         */

        //float lambda = 0.5;
        /*if (rank == 0) {
                printf("using lambda: %.4f\n", lambda);
        }*/

       LLT< MatrixXf> chol;
       MatrixXf L(m,m);

	if (skinny) {
		 /* L = chol(AtA + rho*I) */
                //MatrixXf AtA(n,n);
                //AtA = At*A;
                MatrixXf rhoI(n,n);
                rhoI.setIdentity(n,n);
                L.noalias() = (A.transpose()*A);
		L.noalias() += rhoI*rho;
		chol.compute(L); 
   	}
	else {
                /* L = chol(I + 1/rho*AAt) */
                MatrixXf eye(m,m);
                eye.setIdentity(m,m);
                L.noalias() = (1/rho) * A * A.transpose(); 
                L += eye;
                chol.compute(L);
		
		//llt.compute(eye + (1/rho) * A * A.transpose());
	} 

	/* Main ADMM solver loop */

        int iter=0;
        /*if (rank == 0) {
                printf("%3s %10s %10s %10s %10s %10s\n", "#", "r norm", "eps_pri", "s norm", "eps_dual", "objective");
        }*/

	double time, time2;

        while(iter < MAX_ITER) {

                /* u-update: u = u + x - z */

                u += x - z;

                /* x-update: x = (A^T A + rho I) \ (A^T b + rho z - y) */
                q = Atb+rho*(z - u);

                if (skinny) {
                        /* x = U \ (L \ q) */
                        x = chol.solve(q);
                } else {
                        /* x = q/rho - 1/rho^2 * A^T * (U \ (L \ (A*q))) */
                        //Aq = A * q;
                        //p = chol.solve(Aq);
			x = q/rho -1/(rho*rho) * A.transpose() * chol.solve(A*q); 
	
                        //x = (A.transpose() * chol.solve(A*q)) * (-1/(rho*rho));
                        //q = q * 1/rho;
                        //x = x + q;
                }

                /*
                 * Message-passing: compute the global sum over all processors of the
                 * contents of w and t. Also, update z.
                 */

                w = x + u;

                send[0] = r.transpose()*r; //r * r;
                send[1] = x.transpose()*x; //x * x;
                send[2] = u.transpose()*u  ;
                send[2] /= rho*rho;

		zprev = z;

                // could be reduced to a single Allreduce call by concatenating send to w
		double start;
		start = MPI_Wtime(); 
                MPI_Allreduce(w.data(), z.data(),  n, MPI_FLOAT, MPI_SUM, comm);
                MPI_Allreduce(send,    recv,     3, MPI_FLOAT, MPI_SUM, comm);
		time += MPI_Wtime() - start; 

                prires  = sqrt(recv[0]);  /* sqrt(sum ||r_i||_2^2) */
                nxstack = sqrt(recv[1]);  /* sqrt(sum ||x_i||_2^2) */
                nystack = sqrt(recv[2]);  /* sqrt(sum ||y_i||_2^2) */

                z /= N;

		float k = lambda/(N*rho);
	
		/* soft thresholding inlined */ 
                for (int i=0; i < z.size(); i++) {
                if (z(i) > k)  	    { z(i) = z(i) - k; }
                else if (z(i) < -k) { z(i) = z(i) + k; }
                else                { z(i) = 0; }
                }

                 /* Termination checks */

                /* dual residual */
                zdiff = z - zprev;

                dualres = sqrt(N) * rho * zdiff.squaredNorm();

                /* compute primal and dual feasibility tolerances */
                eps_pri = sqrt(n*N)*ABSTOL + RELTOL * fmax(nxstack, sqrt(N)*z.squaredNorm());
                eps_dual = sqrt(n*N)*ABSTOL + RELTOL * nystack;

                /*if (rank == 0) {
                        printf("%3d %10.4f %10.4f %10.4f %10.4f %10.4f\n", iter,
                                        prires, eps_pri, dualres, eps_dual, objective(A, b, lambda, z));
                }*/

                if (prires <= eps_pri && dualres <= eps_dual) {
                        break;
                }

                 /* Compute residual: r = x - z */
                r = x-z;
		
		iter++; 

        }
	 
	return make_tuple(z, time); 
}


