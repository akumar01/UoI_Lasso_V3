#ifndef MANAGE_DATA_H
#define MANAGE_DATA_H

#ifdef __cplusplus
        extern "C" {
#endif

int get_rows (char *infile, char *dataset);

int get_cols (char *infile, char *dataset);  

float* get_matrix (int Matrows, int Matcols, int Totalrows, MPI_Comm comm, int mpi_rank, char *dataset, char *infile);

float* get_array (int Arrlen, int Totalrows, MPI_Comm comm, int mpi_rank, char *dataset, char *infile);

//void combine_matrix (float * Mat, float *Arr, float * Out, int rows, int cols); 

//float* split_matrix (float * Mat, float *Out, int rows, int cols, int this_rank); 

// void get_train (float * Mat, float * Vec, float * Mat_train, float * Vec_train, int train,  int cols);

//void write_data (char OutFile[], int maxBoot, int bgdOpt, int nrnd, float cvlfrct, float rndfrct, float rndfrctL, int nbootE, int nbootS, int nMP, int seed, int m, int n, 
//		float end_loadTime, float end_distTime, float end_commTime, float end_las1Time, float end_las2Time, float end_olsTime,
//		 float *B0, float *R2m0, float *lamb0, float *B, float *R2m, float *lambL, float *sprt, float *Bgd_m, float *R2_m, int rsd_size_, float *rsd, float *bic, MPI_Comm comm );  


void write_output (char OutFile[], int nrnd, int n, float *Bgd_m, float *R2_m, int rsd_size_, float *rsd, float *bic, MPI_Comm comm ); 

//void write_inter(char OutFile[], int nboot, int n, float *B0, float R20, float lambC, float *B, float R2m, float lambD, float *sprt, MPI_Comm comm ); 


void write_inter(char OutFile[], int nboot, int n, float *B0, float R20, float lambC, float *B, float R2m, float lambD, float *sprt_h, int m, int nbootE, int nrnd, float cvlfrct, float rndfrct, float rndfrctL, float end_read, float tmax, float end_dist, float redis_end, float lasso1_end, float lasso2_end, float est_end, float end_ols, MPI_Comm comm ); 


void write_selections(char OutFile[], float *B, float *R2m, float *lamda, float *sprt_in, int nboot, int nMP, int n, MPI_Comm comm);

void write_results(int bootstraps, int n_lambda, int n_features, float* b_hat, float* bic_scores, char OutFile[], MPI_Comm comm);

void
write_out (int rows, int cols, float *final_result, char OutFile[], MPI_Comm comm, char dataname[]); 

#ifdef __cplusplus
 }
#endif

#endif

