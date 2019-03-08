#define EIGEN_DEFAULT_TO_ROW_MAJOR
#define EIGEN_USE_MKL_ALL
#include <iostream>
#include <fstream>
#include <mpi.h>
#include <eigen3/Eigen/Dense>
#include <mkl.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <tuple>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_statistics.h>
#include "bins.h"
#include "manage-data.h"
#include "structure.h"
#include "CommandLineOptions.h"
#include "lasso.h"

using namespace std;
using namespace Eigen;
float not_NaN (float x) {if (!isnan(x)) return x; else return 0;}

long random_at_mostL(long max) {
  unsigned long num_bins = (unsigned long) max + 1, num_rand = (unsigned long) RAND_MAX + 1, bin_size = num_rand / num_bins, defect = num_rand % num_bins;
  long x;
  do {
    x = random();
  }
  while (num_rand - defect <= (unsigned long)x);
  return x/bin_size;
}

char* readable_fs(float size/*in bytes*/, char *buf) {
  int i = 0;
  const char* units[] = {"B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"};
  while (size > 1024) {
    size /= 1024;
    i++;
  }
  sprintf(buf, "%.*f %s", i, size, units[i]);
  return buf;
}

/*  Only effective if N is much smaller than RAND_MAX */
void shuffle(int *array, size_t n) {
  if (n > 1) {
    size_t i;
    for (i = 0; i < n - 1; i++) {
      size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
      int t = array[j];
      array[j] = array[i];
      array[i] = t;
    }
  }
}

VectorXf logspace (int start, int end, int size) {

  VectorXf vec;
  vec.setLinSpaced(size, start, end);

  for(int i=0; i<size; i++)
    vec(i) = pow(10,vec(i));

  return vec;
}

void print_matrix( MatrixXf m, string name )
{
  std::ofstream file(name);
  if (file.is_open())
  {
    file  << m << '\n';
  }

}

void print_vector( VectorXf m, string name )
{
  std::ofstream file(name);
  if (file.is_open())
  {
    file  << m << '\n';
  }

}

float r2_score (const VectorXf& x, const VectorXf& y)
{
  /*calculates the person R2 values*/
  const float num_observations = static_cast<float>(x.size());
  float x_stddev = sqrt((x.array()-x.mean()).square().sum()/(num_observations-1));
  float y_stddev = sqrt((y.array()-y.mean()).square().sum()/(num_observations-1));
  float numerator = ((x.array() - x.mean() ) * (y.array() - y.mean())).sum() ;
  float denomerator = (num_observations-1)*(x_stddev * y_stddev);
  float r2 = pow((numerator / denomerator),2);
  return r2;
}

float pearson (VectorXf vec1, VectorXf vec2) {
  VectorXd vec1_d = vec1.cast <double>();
  VectorXd vec2_d = vec2.cast <double>();
  gsl_vector_view gsl_x = gsl_vector_view_array( vec1_d.data(), vec1_d.size());
  gsl_vector_view gsl_y = gsl_vector_view_array( vec2_d.data(), vec2_d.size());
  gsl_vector *gsl_v1 =  &gsl_x.vector;
  gsl_vector *gsl_v2 = &gsl_y.vector;
  double r = gsl_stats_correlation (gsl_v1->data, 1, gsl_v2->data, 1, gsl_v1->size);
  float r2 = (float) pow(r, 2);   
  return r2;
}

PermutationMatrix<Dynamic,Dynamic> RandomPermute(int rows)
{
  srand(time(0));
  PermutationMatrix<Dynamic,Dynamic> perm_(rows);
  perm_.setIdentity();
  random_shuffle(perm_.indices().data(), perm_.indices().data()+perm_.indices().size());

  return perm_;
}

MatrixXf CreateSupport(int n_lambdas, int n_bootstraps, int threshold_, MatrixXf estimates)
{
  // creates supports for the estimates from model selection:
  // Input:
  //------------------------------------------------
  // n_lambdas    : int number of lambda parameters
  // n_bootstraps : int number of sel bootstraps. 
  // threshold_   : int used for soft thresholding
  //estimates     : (n_lambda) x (n_bootstraps) x (n_features) 

  //Output:
  //------------------------------------------
  // support      : (n_lambda) x (n_features) support 
  //TODO: support matrix is currently float. Must check compatability and convert it into bool.


  // Note: In Estimates if there is no parallelism, estimates_dense stores lambda values contiguously. 
  // From rows 0-n_lambdas for bootstrap1, n_lambdas+1-2*n_lambdas for bootstrap2 etc.
  // So in CreateSupport function since we reduce across bootstraps for each lambda value we iterate
  // over different lambdas as the outer loop and inner loop in bootstraps and we access each lambda for
  // different bootstraps.


  int n_features = estimates.cols();
  MatrixXf support(n_lambdas, n_features);
  MatrixXi tmp(n_bootstraps, n_features);

  for (int lambda_idx = 0; lambda_idx < n_lambdas; lambda_idx++)
  {
    tmp.setZero();
    for (int bootstraps = 0; bootstraps < n_bootstraps; bootstraps++)
    {
      for (int feature_idx = 0; feature_idx < n_features; feature_idx++)
      {
       if (estimates(((n_lambdas*bootstraps)+lambda_idx), feature_idx) != 0)
          tmp(bootstraps, feature_idx) = 1.0;
        else
          tmp(bootstraps, feature_idx) = 0.0;

      }

    }

    VectorXi sum_v(n_features);

    sum_v = tmp.colwise().sum();

    for (int l = 0; l < n_features; l++)
    {
      if (sum_v(l) >= threshold_)
        support(lambda_idx, l) = 1.0;
      else
        support(lambda_idx, l) = 0.0;

    }

  }

  return support;
}

tuple<MatrixXf, MatrixXf, double, double>
lasso_sweep (MatrixXf X, VectorXf y, VectorXf lambda, float train_frac, int n_bootstraps, bool use_admm, bool debug, int MAX_ITER, float RELTOL, float ABSTOL, float rho, MPI_Comm comm_sweep)
{
  /*
     Perform Lasso regression across bootstraps of a dataset for a sweep
     of L1 penalty values.

     Parameters
     ----------
    X : np.array
    data array containing regressors; assumed to be 2-d array with
    shape n_samples x n_features

    y : np.array
    data array containing dependent variable; assumed to be a 1-d array
    with length n_samples

    lambdas : np.array
    the set of regularization parameters to run boostraps over

    train_frac : float
    float between 0 and 1; the fraction of data to use for training

    n_bootstraps : int
    the number of bootstraps to obtain from the dataset; each bootstrap
    will undergo a Lasso regression

    n_minibatch : int
    number of minibatches to use in case SGD is used for the regression

    use_admm: bool
    switch to use the alternating direction method of multipliers (
    ADMM) algorithm

    Returns
    -------
    estimates : np.array
    predicted regressors for each bootstrap and lambda value; shape is
    (n_bootstraps, n_lambdas, n_features)

    scores : np.array
    scores by the model for each bootstrap and lambda
    value; shape is (n_bootstraps, n_lambdas)
   */

  int rank_sweep;
  MPI_Comm_rank(comm_sweep, &rank_sweep);
  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  //get the shape of X matrix.
  int n_samples = X.rows(); // samples here are samples per core. 
  int n_features = X.cols();
  int n_lambdas = lambda.size();

  //Containers to store the estimates and scores.
  MatrixXf estimates(n_bootstraps * n_lambdas, n_features);
  MatrixXf scores(n_bootstraps, n_lambdas);
  estimates.setZero();
  scores.setZero();


  int n_train_samples = round(train_frac * n_samples);

  /*if ( (rank_sweep == 0) && debug ) 
    {
    cout << " Data ready for UoI " << endl; 
    cout << "n_samples: " << n_samples << " n_train_samples: " << n_train_samples << endl;
    }*/

  //Intermediate containers required for the computation.
  PermutationMatrix<Dynamic,Dynamic> perm(n_samples);
  //MatrixXf X_perm(n_samples, n_features);
  //VectorXf y_perm(n_samples);
  VectorXf y_train(n_train_samples);
  MatrixXf X_train(n_train_samples, n_features);
  VectorXf est_(n_features);
  VectorXf y_hat(n_samples-n_train_samples);
  VectorXf y_true(n_samples-n_train_samples);
  VectorXf y_test(n_samples-n_train_samples);
  MatrixXf X_test(n_samples-n_train_samples, n_features);

  double time1, time2, time3, time4,small_red;

  for (int bootstrap=0; bootstrap < n_bootstraps; bootstrap++)
  {
    perm = RandomPermute(n_samples);

    //Random Shuffle X and y
    X = perm * X;
    y = perm * y;

    //Split X and y into train and test dataset.
    y_train = y.head(n_train_samples);
    X_train = X.topRows(n_train_samples);
    X_test = X.bottomRows(n_samples-n_train_samples);
    y_test = y.tail(n_samples-n_train_samples);

    if (rank_sweep == 0 && bootstrap == 0 && debug)
    {
      print_vector( lambda, "./debug/lambda" + to_string(rank) + ".txt");
      print_matrix( X_train, "./debug/X_train.txt");
      print_matrix( X_test, "./debug/X_test.txt");
      print_vector( y_train, "./debug/y_train.txt");
      print_vector( y_test, "./debug/y_test.txt");

    }

    for (int lambda_idx=0; lambda_idx < n_lambdas; lambda_idx++)
    {
      //if (rank_sweep==0)
      //      cout << "top lambda_idx: " << lambda_idx << endl;
      float n_lamb = lambda(lambda_idx);
      if ( rank_sweep == 0 )
        time1 = MPI_Wtime();

      tie(est_,time3) = lasso(X_train, y_train.array()-y_train.mean(), n_lamb, MAX_ITER, RELTOL, ABSTOL, rho, comm_sweep);

      //if(rank_sweep==0) {
      //	cout << "time for 1 z all reduce: " << time3 << endl;
	    //    cout << "time for 1 w all reduce: " << small_red << endl;
	    //    MPI_Abort(comm_sweep, 23);
      //} 

      if(rank_sweep==0)
      {
        time2 += MPI_Wtime() - time1;
        time4 += time3;
        //cout << "time for 1 lasso: " << time2 << "(s)" << endl;
        //cout << "time for 1 lasso comm: " << time4 << "(s)" << endl;
        //cout << "time for 1 lasso comp: " << time2-time4 << "(s)" << endl;
      }
      estimates.row((bootstrap*n_lambdas)+lambda_idx) = est_.unaryExpr(ptr_fun(not_NaN)); //this stores estimates from 0-n_lambdas consecutively -- useful for parallel computing.
      //estimates.row((lambda_idx*n_bootstraps) + bootstrap) = est_.unaryExpr(ptr_fun(not_NaN)); // this stores estimates in order 0 in n_lambda steps. so bootstraps are strored consecutively.
      y_hat = X_test * est_;
      y_true = y_test.array()-y_test.mean();
      scores(bootstrap, lambda_idx) = pearson(y_hat, y_true);
      //scores(bootstrap, lambda_idx) = r2_score(y_hat, y_true);

      /*if (rank_sweep==0)
        {
        cout << "Score --> " << scores(bootstrap, lambda_idx) << endl;
      //MPI_Abort(comm_sweep, 23);
      }*/

      if (rank_sweep == 0 && bootstrap == 0 && lambda_idx == 0 && debug)
      {
        print_vector( est_, "./debug/est_dense_0_0.txt");
        print_vector( y_hat, "./debug/y_hat_0_0.txt");
        print_vector( y_true, "./debug/y_true_0_0.txt");
        print_matrix( scores, "./debug/scores_0_0.txt");
      }
    }
  }

  return make_tuple(estimates, scores, time2, time4);

}

tuple<MatrixXf,vector<int> > ApplySupport(MatrixXf& H, VectorXf support)
{

  vector<int> ids;

  for(int i = 0; i<support.size(); i++)
    if (support(i) !=0 )
      ids.push_back(i);
  MatrixXf ret(H.rows(), ids.size());

  for(int i=0; i<ids.size(); i++)
    ret.col(i) = H.col(ids[i]);

  return make_tuple(ret,ids);

}

inline float BIC(float n_features, float n_samples, float rss)
{
  /*
     Calculate the Bayesian Information Criterion under the assumption of
     normally distributed disturbances (which allows the BIC to take on the
     simple form below).

     Parameters
     ----------
    n_features : int
    number of model features

    n_samples : int
    number of samples in the dataset

    rss : float
    the residual sum of squares

    Returns
    -------
    BIC : float
    Bayesian Information Criterion
   */

  float bic = -n_samples * log(rss/n_samples) - n_features * log(n_samples);
  return bic;


}

VectorXf median (MatrixXf mat) {

  VectorXf get_col;
  vector<float> v;

  for (int i=0; i<mat.cols(); i++) {
    get_col = mat.col(i);
    nth_element (get_col.data(), get_col.data()+ get_col.size()/2,
                      get_col.data()+get_col.size());

    v.push_back(get_col(get_col.size()/2));

  }

  Map<VectorXf> vec (v.data(), v.size());

  return vec;
}

int main(int argc, char** argv) {

  int rank, nprocs;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  double tic, toc;
  tic = MPI_Wtime();

  char buf[20];

  int nrows;  // A( nrows X ncols )
  int ncols;
  int krows;   // B( krows X ncols )
  //int lasso_init.n_groups; // number of worker groups
  int f,s;
  int j;

  srand(rank);


  INIT lasso_init;

  CommandLineOptions opts;
  string version = "02.00.000";

  CommandLineOptions::statusReturn_e temp = opts.parse( argc, argv );
  lasso_init.debug = opts.getDebug();
  string InputFile = opts.getInputFile();
  string OutputFile1 = opts.getOutputFile1();
  string OutputFile2 = opts.getOutputFile2();
  string data_matrix = opts.getDatasetMatrix();
  string data_vector = opts.getDatasetVector();
  lasso_init.Infile = new char[InputFile.length() + 1];
  lasso_init.Outfile1 = new char [OutputFile1.length() + 1];
  lasso_init.Outfile2 = new char [OutputFile2.length() + 1];
  lasso_init.data_mat = new char [data_matrix.length() + 1];
  lasso_init.data_vec = new char [data_vector.length() + 1];
  strcpy(lasso_init.Infile, InputFile.c_str());
  strcpy(lasso_init.Outfile1, OutputFile1.c_str());
  strcpy(lasso_init.Outfile2, OutputFile2.c_str());
  strcpy(lasso_init.data_mat, data_matrix.c_str());
  strcpy(lasso_init.data_vec, data_vector.c_str());
  lasso_init.verbose =  opts.getVerbose();

  if (rank == 0)
  {


    lasso_init.n_lambdas = opts.getLambdas();
    lasso_init.selection_thres_frac = opts.getSelectionThreshold();
    lasso_init.train_frac_sel = opts.getTrainSelection();
    lasso_init.train_frac_est = opts.getTrainEstimation();
    lasso_init.train_frac_overall = opts.getTrainOverall();
    lasso_init.n_boots_coarse = opts.getBootsCoarse();
    lasso_init.n_boots_sel = opts.getBootsSel();
    lasso_init.n_boots_est = opts.getBootsEst();
    lasso_init.bagging_options = opts.getBaggingOption();
    lasso_init.n_groups = opts.getnGroups();
    lasso_init.n_minigroups = opts.getMiniGroups();
    lasso_init.n_coarse = opts.getnCoarse();
    lasso_init.n_minicoarse = opts.getnMiniCoarse();
    lasso_init.max_iter = opts.getMAXITER();
    lasso_init.reltol = opts.getRELTOL();
    lasso_init.abstol = opts.getABSTOL();
    lasso_init.rho = opts.getRho();
  }

  //send all the intialized variables from rank 0  to other processes
  MPI_Bcast(&lasso_init.n_lambdas, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.selection_thres_frac, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.train_frac_sel, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.train_frac_est, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.train_frac_overall, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.n_boots_coarse, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.n_boots_sel, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.n_boots_est, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.bagging_options, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.n_groups, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.n_minigroups, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.n_coarse, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.n_minicoarse, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.max_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.reltol, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.abstol, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&lasso_init.rho, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
 

  if (rank == 0) {
    nrows = get_rows(lasso_init.Infile,lasso_init.data_mat);
    ncols = get_cols(lasso_init.Infile,lasso_init.data_mat);

  }

  MPI_Bcast(&nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&ncols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {

    size_t sizeA = (size_t) nrows * (size_t) ncols * sizeof(float);
    size_t sizeB = (size_t) lasso_init.n_groups * (size_t) nrows * (size_t) ncols * sizeof(float);

    printf("Total A: %s\n", readable_fs((float) sizeA, buf));
    printf("Total B: %s\n", readable_fs((float) sizeB, buf));
    printf("Total:   %s\n", readable_fs((float) (sizeA+sizeB), buf));

    printf("A per rank: %s\n", readable_fs( (float) sizeA / (double) nprocs , buf));
    printf("B per rank: %s\n", readable_fs( (float) sizeB / (double) nprocs , buf));

    printf("Num procs: %i\n", nprocs);
    printf("B groups: %i\n\n", lasso_init.n_groups);
    printf("A dimensions: (%i, %i)\n", nrows, ncols);
    printf("B dimensions: (%i, %i)\n", nrows, ncols);

    if ( nprocs > nrows ) {
      printf("must have nprocs < nrows \n");
      fflush(stdout);
      MPI_Abort(MPI_COMM_WORLD, 3);
    }

    if ( lasso_init.n_groups > nprocs ) {
      printf("must have ngroups < nprocs \n");
      fflush(stdout);
      MPI_Abort(MPI_COMM_WORLD, 4);
    }   

  }

  /*MPI_Bcast(&nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ncols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&krows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&lasso_init.n_groups, 1, MPI_INT, 0, MPI_COMM_WORLD);*/

  double load_t = MPI_Wtime();
  if(rank==0) cout << "(1) Preparing to load data in parallel." << endl;

  int local_rows = bin_size_1D(rank, nrows, nprocs);

  size_t sizeX = (size_t) local_rows * (size_t) ncols * sizeof(float);  
  float *X;
  X = (float *) malloc(sizeX);
  X = get_matrix(local_rows, ncols, nrows, MPI_COMM_WORLD, rank, lasso_init.data_mat, lasso_init.Infile); 
  Map<MatrixXf> X_(X, local_rows, ncols);

  float *y;
  y = (float*) malloc(ncols * sizeof(float));
  y = get_array(local_rows, nrows, MPI_COMM_WORLD, rank, lasso_init.data_vec, lasso_init.Infile);
  Map<VectorXf> y_(y, local_rows); 
  //memset(A, rank, sizeA);

  if (rank==0 && lasso_init.verbose) {
	cout << "Data loaded in " << MPI_Wtime() - load_t << "(s)" << endl;
  }

  if(rank==0 && lasso_init.debug) {
    print_vector(y_,"./debug/y.txt" ); 
  }

  double pre_t = MPI_Wtime();

  int qcols = ncols+1;
  MatrixXf A_(local_rows, qcols);
  A_ << X_,y_;
  free(X);
  free(y);
  float *A;
  size_t sizeA = (size_t) local_rows * (size_t) qcols * sizeof(float);
  A = (float*) malloc(sizeA);
  Map<MatrixXf>(A, local_rows, qcols) = A_; 

  int color = bin_coord_1D(rank, nprocs, lasso_init.n_coarse);  
  MPI_Comm comm_c;
  MPI_Comm_split(MPI_COMM_WORLD, color, rank, &comm_c);

  int nprocs_c, rank_c;
  MPI_Comm_size(comm_c, &nprocs_c);
  MPI_Comm_rank(comm_c, &rank_c);
  MPI_Group n_coarse_group;
  MPI_Comm_group(comm_c, &n_coarse_group);
  //printf ("(RANK_G/SIZE_G) = (%d/%d)\n", rank_c, nprocs_c);  

  if (rank==0) cout << "Preprocessing time: " << MPI_Wtime() - pre_t << "(s)" << endl;

  int qrows = bin_size_1D(rank_c, nrows, nprocs_c);
  size_t sizeB = (size_t) qcols * (size_t) qrows * sizeof(float);   
  float *B; 
  B = (float *) malloc(sizeB);

  {

    MPI_Win win;
    MPI_Win_create(A, sizeA, sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
#ifndef SIMPLESAMPLE
    int *sample;
    if (rank_c == 0) {
      sample = (int*)malloc( nrows * sizeof(int) );
      for (int i=0; i<nrows; i++) sample[i]=i;
      shuffle(sample, nrows);
    } else {
      sample = NULL;
    }

    int srows[qrows];

    {
      int sendcounts[nprocs_c];
      int displs[nprocs_c];

      for (int i=0; i<nprocs_c; i++) {
        int ubound;
        bin_range_1D(i, nrows, nprocs_c, &displs[i], &ubound);
        sendcounts[i] = bin_size_1D(i, nrows, nprocs_c);
      }

      MPI_Scatterv(sample, sendcounts, displs, MPI_INT, &srows, qrows, MPI_INT, 0, comm_c);

      if (rank_c == 0) free(sample);
    }
#endif

    double t = MPI_Wtime();
    MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, win);

    for (int i=0; i<qrows; i++) {
#ifdef SIMPLESAMPLE
      int trow = (int) random_at_mostL( (long) nrows);
#else
      int trow = srows[i];
#endif
      int target_rank = bin_coord_1D(trow, nrows, nprocs);
      int target_disp = bin_index_1D(trow, nrows, nprocs) * qcols;
      MPI_Get( &B[i*qcols], qcols, MPI_FLOAT, target_rank, target_disp, qcols, MPI_FLOAT, win);
    }

    MPI_Win_fence(MPI_MODE_NOSUCCEED, win);

    double tmax, tcomm = MPI_Wtime() - t;
    MPI_Reduce(&tcomm, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      printf("Comm time: %f (s)\n", tmax);
    }

    MPI_Win_free(&win);

  }
  //free(A);
  /* do work on B here */
  Map<MatrixXf> B_(B, qrows, qcols);

  if (rank==0 && lasso_init.debug) {
    print_matrix(B_, "./debug/B.txt");
  }

  MatrixXf X_coarse(local_rows, ncols);
  X_coarse = B_.block(0, 0, B_.rows(), B_.cols()-1);
  VectorXf y_coarse(local_rows);
  y_coarse = B_.rightCols(1); 

  free(B); 

  /*Create all communicators*/
  int color1 = bin_coord_1D(rank_c, nprocs_c, lasso_init.n_minicoarse);
  MPI_Comm comm_mc;
  MPI_Comm_split(comm_c, color1, rank_c, &comm_mc);

  int nprocs_mc, rank_mc;
  MPI_Comm_size(comm_mc, &nprocs_mc);
  MPI_Comm_rank(comm_mc, &rank_mc);

  //Level-1 root ranks group creation.
  int cboot_root_rank = -1, cboot_root_size = -1;

  VectorXi cboot_roots(lasso_init.n_coarse);
  int root = 0;

  for (int i=0; i<lasso_init.n_coarse; i++)
  {
    root = i * nprocs_c;
    cboot_roots(i) = root;
  }

  if(rank==0 && lasso_init.debug) cout << "before n_coarse include" << endl;

  MPI_Group cboot_root_group;
  MPI_Group_incl(world_group, lasso_init.n_coarse, cboot_roots.data(), &cboot_root_group);
  MPI_Comm cboot_roots_comm; 

  MPI_Comm_create_group(MPI_COMM_WORLD, cboot_root_group, 0, &cboot_roots_comm);

  if(rank==0 && lasso_init.debug) cout << "after n_coarse include" << endl;

  if (MPI_COMM_NULL != cboot_roots_comm) {  
    MPI_Comm_rank(cboot_roots_comm, &cboot_root_rank);
    MPI_Comm_size(cboot_roots_comm, &cboot_root_size);
  }

  //Level-2 root ranks group creation.
  int clam_root_rank = -1, clam_root_size = -1;

  VectorXi clam_roots(lasso_init.n_minicoarse);
  root = 0;

  for (int i=0; i<lasso_init.n_minicoarse; i++) {
    root = i * nprocs_mc;
    clam_roots(i) = root;
  }

  if(rank==0 && lasso_init.debug) cout << "before n_minicoarse include" << endl;

  MPI_Group clam_root_group;
  MPI_Group_incl(cboot_root_group, lasso_init.n_minicoarse, clam_roots.data(), &clam_root_group);
  MPI_Comm clam_roots_comm;
  MPI_Comm_create_group(comm_c, clam_root_group, 0, &clam_roots_comm);

  if(rank==0 && lasso_init.debug) cout << "after n_coarse include" << endl;

  if (MPI_COMM_NULL != clam_roots_comm)  {
    MPI_Comm_rank(clam_roots_comm, &clam_root_rank);
    MPI_Comm_size(clam_roots_comm, &clam_root_size);
  }


  //------------------------------------------------
  //create color with init.n_groups for level 1 parallelization
  int color2 = bin_coord_1D(rank, nprocs, lasso_init.n_groups);
  MPI_Comm comm_g;
  MPI_Comm_split(MPI_COMM_WORLD, color2, rank, &comm_g);

  int nprocs_g, rank_g;
  MPI_Comm_size(comm_g, &nprocs_g);
  MPI_Comm_rank(comm_g, &rank_g);
  MPI_Group L1_group;
  MPI_Comm_group(comm_g, &L1_group);

  //create color iwth init.n_minigroups for level 2 parallelism
  int color3 = bin_coord_1D(rank_g, nprocs_g, lasso_init.n_minigroups);
  MPI_Comm comm_mg;
  MPI_Comm_split(comm_g, color3, rank_g, &comm_mg);

  int nprocs_mg, rank_mg;
  MPI_Comm_size(comm_mg, &nprocs_mg);
  MPI_Comm_rank(comm_mg, &rank_mg);

  if(rank==0 && lasso_init.debug) {
    cout << "nprocs_g: " << nprocs_g << endl;
    cout << "nprocs_mg:" << nprocs_mg << endl;
  }

  //create L1 and L2 root processes as separate groups.
  //This is for easier output management.

  //Level-1 root ranks group creation.
  int l1_root_rank = -1, l1_root_size = -1;

  VectorXi l1_roots(lasso_init.n_groups);
  root = 0;

  for (int i=0; i<lasso_init.n_groups; i++)
  {
    root = i * nprocs_g;
    l1_roots(i) = root;
  }

  if(rank==0 && lasso_init.debug) cout << "before n_group include" << endl;

  MPI_Group l1_root_group;
  MPI_Group_incl(world_group, lasso_init.n_groups, l1_roots.data(), &l1_root_group);
  MPI_Comm L1_roots_comm;
  MPI_Comm_create_group(MPI_COMM_WORLD, l1_root_group, 0, &L1_roots_comm);

  if(rank==0 && lasso_init.debug) cout << "after n_group include" << endl;

  if (MPI_COMM_NULL != L1_roots_comm)
  {
    MPI_Comm_rank(L1_roots_comm, &l1_root_rank);
    MPI_Comm_size(L1_roots_comm, &l1_root_size);
  }

  //Level-2 root ranks group creation.
  int l2_root_rank = -1, l2_root_size = -1;

  VectorXi l2_roots(lasso_init.n_minigroups);
  root = 0;

  for (int i=0; i<lasso_init.n_minigroups; i++)
  {
    root = i * nprocs_mg;
    l2_roots(i) = root;
  }

  if(rank==0 && lasso_init.debug) cout << "before n_minigroup include" << endl;

  MPI_Group l2_root_group;
  MPI_Group_incl(L1_group, lasso_init.n_minigroups, l2_roots.data(), &l2_root_group);
  MPI_Comm L2_roots_comm;
  MPI_Comm_create_group(comm_g, l2_root_group, 0, &L2_roots_comm);

  if(rank==0 && lasso_init.debug) cout << "after n_minigroup include" << endl;

  if (MPI_COMM_NULL != L2_roots_comm)
  {
    MPI_Comm_rank(L2_roots_comm, &l2_root_rank);
    MPI_Comm_size(L2_roots_comm, &l2_root_size);
  }

  //Completed creating all logical processes split communicators

  VectorXf lambda_coarse(lasso_init.n_lambdas);
  VectorXf lambda_coarse_dis(lasso_init.n_lambdas/lasso_init.n_minicoarse);

  if (rank_c == 0 )
  {
    if (lasso_init.n_lambdas == 1)
      lambda_coarse.setOnes();
    else
      lambda_coarse = logspace(-3, 3, lasso_init.n_lambdas);
  }

  MPI_Bcast(lambda_coarse.data(), lambda_coarse.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);

  if (MPI_COMM_NULL != clam_roots_comm) {
    lambda_coarse_dis = lambda_coarse.segment(clam_root_rank*(lasso_init.n_lambdas/lasso_init.n_minicoarse), lasso_init.n_lambdas/lasso_init.n_minicoarse);

  }

  MPI_Bcast(lambda_coarse_dis.data(), lambda_coarse_dis.size(), MPI_FLOAT, 0, comm_mc);

  if ( rank == 0 )
    cout << "lambda coarse created for " << lambda_coarse.size() << " size." << endl;

  MatrixXf estimates_coarse, scores_coarse;
  double time1, time2;
  double coarse_lass, coarse_lass_comm;
  //run the coarse lasso sweep    

  if (rank == 0)
    time1 = MPI_Wtime();

  tie(estimates_coarse, scores_coarse, coarse_lass, coarse_lass_comm)  =
    lasso_sweep (X_coarse, y_coarse, lambda_coarse_dis, lasso_init.train_frac_sel, lasso_init.n_boots_coarse/lasso_init.n_coarse, lasso_init.use_admm, lasso_init.debug, lasso_init.max_iter, lasso_init.reltol, lasso_init.abstol, lasso_init.rho, comm_mc);

  if (rank==0)
	time2 = MPI_Wtime() - time1;
  //free matrix
  X_coarse.resize(0,0);
  y_coarse.setZero();

  //---------------------------------------------------------------
  //Collect data 
  double g_time1;
  g_time1 = MPI_Wtime();

  MatrixXf estimates_cboot_root, scores_cboot_root, scores_clam_root, scores_mc;

  if (rank == 0) {
    estimates_cboot_root.setZero(lasso_init.n_boots_coarse*lasso_init.n_lambdas, ncols);
    scores_cboot_root.setZero(lasso_init.n_boots_coarse, lasso_init.n_lambdas);
  }

  if (rank_c == 0) 
    scores_clam_root.setZero(lasso_init.n_boots_coarse/lasso_init.n_coarse, lasso_init.n_lambdas);

  if(rank_mc==0)
    scores_mc.setZero(scores_coarse.rows(), scores_coarse.cols());
  


  MPI_Reduce(scores_coarse.data(), scores_mc.data(), scores_coarse.rows() * scores_coarse.cols(), MPI_FLOAT, MPI_SUM, 0, comm_mc);

  if (rank==0 && lasso_init.debug) cout << "Finished 1st reduce" << endl;

  if (rank_mc==0)
    scores_mc /= nprocs_mc;

  if (rank==0 && lasso_init.debug) cout << "Finished the division" << endl;


  if (MPI_COMM_NULL != clam_roots_comm)
    MPI_Gather(scores_mc.data(), scores_mc.rows() * scores_mc.cols(), MPI_FLOAT,
        scores_clam_root.data(), scores_mc.rows() * scores_mc.cols(), MPI_FLOAT, 0, clam_roots_comm);

  if (rank==0 && lasso_init.debug) cout << "Completed lambda level gather" << endl;

  if (MPI_COMM_NULL != cboot_roots_comm) {
    MPI_Gather(scores_clam_root.data(), scores_clam_root.rows() * scores_clam_root.cols(), MPI_FLOAT,
        scores_cboot_root.data(), scores_clam_root.rows() * scores_clam_root.cols(), MPI_FLOAT, 0, cboot_roots_comm);

    MPI_Gather(estimates_coarse.data(), estimates_cboot_root.rows() * estimates_cboot_root.cols(), MPI_FLOAT,
        estimates_cboot_root.data(), estimates_cboot_root.rows() * estimates_cboot_root.cols(), MPI_FLOAT, 0, cboot_roots_comm);
  }

  //-------------------------------------------------------------

  if (rank==0 && lasso_init.debug) {
    cout << "All gathers complete" << endl;
  }

  if ((rank == 0) && lasso_init.verbose) {
    cout << "Coarse sweep time: " << time2 << "(s)" << endl;
    cout << "Lasso Comm time:" << coarse_lass_comm << "(s)" << endl;
    cout << "Lasso Comp time:" << coarse_lass-coarse_lass_comm << endl;
    cout << "Results gather time:" << MPI_Wtime()-g_time1 << endl;
  }

  if ( (rank == 0) && lasso_init.debug) {
    print_matrix( estimates_coarse, "./debug/estimated_coarse.txt");
    print_matrix( scores_coarse, "./debug/scores_coarse.txt");
  }

  //empty the unneccesary martices.
  estimates_coarse.resize(0,0);
  scores_coarse.resize(0,0);

  //deduce the index which maximizes the explained variance over bootstraps
  float d_lambda, lambda_max;
  if ( rank == 0 ) {
    VectorXf mean;
    mean = scores_cboot_root.colwise().mean();
    VectorXf::Index lambda_max_idx;

    float max = mean.maxCoeff( &lambda_max_idx );

    //obtain the lambda which maximizes the explained variance over bootstraps
    lambda_max = lambda_coarse(lambda_max_idx);

    //in our dense sweep, we'll explore lambda values which encompass a
    //range that's one order of magnitude less than lambda_max itself
    d_lambda = pow(10, floor(log10(lambda_max)-1));
  }

  if ( (rank == 0) && lasso_init.debug) {
    //print_vector( mean, "./debug/mean_vector.txt");
    print_vector( lambda_coarse, "./debug/lambda_coarse.txt");
    cout << "lambda_max: " << lambda_max << endl;

  }

  //now that we've narrowed down the regularization parameters,
  //we'll run a dense sweep which begins the model selection module of UoI

  //#######################
  //### Model Selection ###
  // #######################
  MPI_Barrier(MPI_COMM_WORLD);

  double sel_time_s, sel_time_e;

  if ( (rank == 0) && lasso_init.verbose)
    cout << "(2) Beginning model selection. Exploring penalty region centered at " << lambda_max << "." << endl;

  VectorXf lambdas(lasso_init.n_lambdas);
  VectorXf lambda_dis(lasso_init.n_lambdas/lasso_init.n_minigroups );

  if (rank==0 ) {
    lambdas.setZero(lasso_init.n_lambdas);

    if (lasso_init.n_lambdas == 1)
      lambdas << lambda_max;
    else
      lambdas.setLinSpaced(lasso_init.n_lambdas, lambda_max - 5 * d_lambda, lambda_max + 5 * d_lambda);
  }

  //if(MPI_COMM_NULL != L1_roots_comm)
   MPI_Bcast(lambdas.data(), lasso_init.n_lambdas, MPI_FLOAT, 0, MPI_COMM_WORLD);


  /*if (MPI_COMM_NULL != L2_roots_comm)
    MPI_Scatter(lambdas.data(), lasso_init.n_lambdas/lasso_init.n_minigroups, MPI_FLOAT, lambda_dis.data(), lasso_init.n_lambdas/lasso_init.n_minigroups, MPI_FLOAT, 0, L2_roots_comm); */

  if (MPI_COMM_NULL != L2_roots_comm)
    lambda_dis = lambdas.segment(l2_root_rank*(lasso_init.n_lambdas/lasso_init.n_minigroups), lasso_init.n_lambdas/lasso_init.n_minigroups);


  //if (MPI_COMM_NULL != L2_roots_comm)
  MPI_Bcast(lambda_dis.data(), lambda_dis.size(), MPI_FLOAT, 0, comm_mg);

  qrows = bin_size_1D(rank_g, nrows, nprocs_g) ;

  float *B__;
  B__ = (float *) malloc( qrows * qcols * sizeof(float) );

  //second randomization for selection

  { 
    MPI_Win win;
    MPI_Win_create(A, sizeA, sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

#ifndef SIMPLESAMPLE
    int *sample;
    if (rank_g == 0) {
      sample = (int*)malloc( nrows * sizeof(int) );
      for (int i=0; i<nrows; i++) sample[i]=i;
      shuffle(sample, nrows);
    } else {
      sample = NULL;
    }

    int srows[qrows];

    {
      int sendcounts[nprocs_g];
      int displs[nprocs_g];

      for (int i=0; i<nprocs_g; i++) {
        int ubound;
        bin_range_1D(i, nrows, nprocs_g, &displs[i], &ubound);
        sendcounts[i] = bin_size_1D(i, nrows, nprocs_g);
      }

      MPI_Scatterv(sample, sendcounts, displs, MPI_INT, &srows, qrows, MPI_INT, 0, comm_g);

      if (rank_g==0) free(sample);
    }

#endif

    double t = MPI_Wtime();
    MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, win);

    for (int i=0; i<qrows; i++) {
#ifdef SIMPLESAMPLE
      int trow = (int) random_at_mostL( (long) nrows);
#else
      int trow = srows[i];
#endif
      int target_rank = bin_coord_1D(trow, nrows, nprocs);
      int target_disp = bin_index_1D(trow, nrows, nprocs) * qcols;
      MPI_Get( &B__[i*qcols], qcols, MPI_FLOAT, target_rank, target_disp, qcols, MPI_FLOAT, win);
    }

    MPI_Win_fence(MPI_MODE_NOSUCCEED, win);

    double tmax, tcomm = MPI_Wtime() - t;
    MPI_Reduce(&tcomm, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      printf("Comm time: %f (s)\n", tmax);
    }

    MPI_Win_free(&win);

  }

  Map<MatrixXf> B_sel( B__, qrows, qcols);
  MatrixXf X_sel(local_rows, ncols);

  X_sel = B_sel.block(0, 0, B_sel.rows(), B_sel.cols()-1);
  y_coarse = B_sel.rightCols(1);
  double lass_time, lass_comm;
  free(B__);

  MatrixXf estimates_dense, scores_dense;
  if (rank == 0)
    sel_time_s = MPI_Wtime(); 

  tie (estimates_dense, scores_dense,lass_time, lass_comm)  =
    lasso_sweep (X_sel, y_coarse, lambda_dis, lasso_init.train_frac_sel, (lasso_init.n_boots_sel/lasso_init.n_groups),
        lasso_init.use_admm, lasso_init.debug, lasso_init.max_iter, lasso_init.reltol, lasso_init.abstol,
        lasso_init.rho, comm_mg);

  //free matrices
  X_sel.resize(0,0);
  y_coarse.setZero();
  //y_coarse.resize(0,0);

  double sup_time_s, sup_time_e;

  if (rank == 0) {
    sel_time_e = MPI_Wtime() - sel_time_s;
    sup_time_s = MPI_Wtime();
  }

  if ( rank == 0 &&  lasso_init.verbose ) {
    cout << "Total Selection lasso time: " << sel_time_e << "(s)" << endl;
    cout << "Total Lasso communication time: " << lass_comm << "(s)" << endl;
    cout << "Total Lasso computation time: " << lass_time-lass_comm << "(s)" << endl;
  }

  if ( (rank == 0) && lasso_init.debug) {
    print_matrix( estimates_dense, "./debug/estimated_dense.txt");
    print_matrix( scores_dense, "./debug/scores_dense.txt");
  }

  double gath_t_sel, gath_t_sel_e;
  
  if(rank==0) gath_t_sel = MPI_Wtime();

  MatrixXf estimates_l2, estimates_l1;

  if(rank_g == 0)
    estimates_l2.setZero((lasso_init.n_boots_sel/lasso_init.n_groups)*lasso_init.n_lambdas, ncols);

  if(rank==0)
    estimates_l1.setZero(lasso_init.n_boots_sel*lasso_init.n_lambdas, ncols);

  if(MPI_COMM_NULL != L2_roots_comm)
     MPI_Gather(estimates_dense.data(), estimates_dense.rows() * estimates_dense.cols(), MPI_FLOAT,
        estimates_l2.data(), estimates_dense.rows() * estimates_dense.cols(), MPI_FLOAT, 0, L2_roots_comm);

  if(MPI_COMM_NULL != L1_roots_comm)
     MPI_Gather(estimates_l2.data(), estimates_l2.rows() * estimates_l2.cols(), MPI_FLOAT,
        estimates_l1.data(), estimates_l2.rows() * estimates_l2.cols(), MPI_FLOAT, 0, L1_roots_comm);

  if(rank==0 && lasso_init.verbose)
    cout << "Selection Gather time: " << MPI_Wtime() - gath_t_sel << "(s)" << endl;
 
 
  //#########################
  //### Supports Creation ###
  //#########################

  //intersect supports across bootstraps for each lambda value
  //we impose a (potentially) soft intersection

  int threshold = lasso_init.selection_thres_frac * lasso_init.n_boots_sel;

  //create support matrix storage
  MatrixXf supports_mtx, supports_(lasso_init.n_lambdas/lasso_init.n_minigroups, ncols); 
  
  if(rank_g == 0)
     supports_mtx.setZero(lasso_init.n_lambdas, ncols);

  if ( rank == 0)
    supports_mtx = CreateSupport(lasso_init.n_lambdas, lasso_init.n_boots_sel, threshold, estimates_l1);

  if (rank == 0)
    sup_time_e = MPI_Wtime() - sup_time_s;

  if ( (rank == 0) && lasso_init.debug)
    print_matrix( supports_mtx, "./debug/supports_.txt");

  double sup_b_time;

  if(rank==0) sup_b_time = MPI_Wtime();

  if (MPI_COMM_NULL != L1_roots_comm)
    MPI_Bcast(supports_mtx.data(), lasso_init.n_lambdas * ncols, MPI_FLOAT, 0, L1_roots_comm);

  if (MPI_COMM_NULL != L2_roots_comm)
    MPI_Scatter(supports_mtx.data(), (lasso_init.n_lambdas/lasso_init.n_minigroups) * ncols, MPI_FLOAT,
			supports_.data(), (lasso_init.n_lambdas/lasso_init.n_minigroups) * ncols, MPI_FLOAT, 0, L2_roots_comm); 

  MPI_Bcast(supports_.data(), supports_.rows() * supports_.cols(), MPI_FLOAT, 0, comm_mg);


  MPI_Barrier(MPI_COMM_WORLD);

  //#######################
  //### Model Estimation ###
  // #######################

  //we'll use the supports obtained in the selection module to calculate
  //bagged OLS estimates over bootstraps

  if ( (rank == 0 ) && lasso_init.verbose )
  {
    cout << "(3) Model selection complete. " << endl;
    cout << "Model Selection time: " << sel_time_e << "(s)" << endl;
    cout << "Support creation time: " << sup_time_e << "(s)"<< endl;
    cout << "Support bcast time: " << MPI_Wtime() - sup_b_time << "(s)" << endl; 
    cout << "(4) Beginning model estimation, with " << lasso_init.n_boots_est << " bootstraps" << endl;
  }

  float *_B_;
  _B_ = (float *) malloc( qrows * qcols * sizeof(float) );

  //Distribute data again for cross-validation step.
  {
    MPI_Win win;
    MPI_Win_create(A, sizeA, sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

#ifndef SIMPLESAMPLE
    int *sample;
    if (rank_g == 0) {
      sample = (int*)malloc( nrows * sizeof(int) );
      for (int i=0; i<nrows; i++) sample[i]=i;
      shuffle(sample, nrows);
    } else {
      sample = NULL;
    }

    int srows[qrows];

    {
      int sendcounts[nprocs_g];
      int displs[nprocs_g];

      for (int i=0; i<nprocs_g; i++) {
        int ubound;
        bin_range_1D(i, nrows, nprocs_g, &displs[i], &ubound);
        sendcounts[i] = bin_size_1D(i, nrows, nprocs_g);
      }

      MPI_Scatterv(sample, sendcounts, displs, MPI_INT, &srows, qrows, MPI_INT, 0, comm_g);

      if (rank_g==0) free(sample);
    }

#endif

    double t = MPI_Wtime();
    MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, win);

    for (int i=0; i<qrows; i++) {
#ifdef SIMPLESAMPLE
      int trow = (int) random_at_mostL( (long) nrows);
#else
      int trow = srows[i];
#endif
      int target_rank = bin_coord_1D(trow, nrows, nprocs);
      int target_disp = bin_index_1D(trow, nrows, nprocs) * qcols;
      MPI_Get( &_B_[i*qcols], qcols, MPI_FLOAT, target_rank, target_disp, qcols, MPI_FLOAT, win);
    }

    MPI_Win_fence(MPI_MODE_NOSUCCEED, win);

    double tmax, tcomm = MPI_Wtime() - t;
    MPI_Reduce(&tcomm, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      printf("Comm time: %f (s)\n", tmax);
    }

    MPI_Win_free(&win);
  }

  free(A);
  Map<MatrixXf> B_est( _B_, qrows, qcols);
  MatrixXf X_est(B_est.rows(), B_est.cols()-1);
  X_est = B_est.block(0, 0, B_est.rows(), B_est.cols()-1);
  VectorXf y_est(B_est.rows());
  y_est = B_est.rightCols(1);
  free(_B_);

  //create or overwrite arrays to collect final results
  VectorXf   coef_(ncols);
 
  float scores_;
  //determine how many samples will be used for overall training
  int train_split = round( lasso_init.train_frac_overall * qrows);
  //determine how many samples will be used for training within a bootstrap       
  int boot_train_split = round( lasso_init.train_frac_est * train_split);

  //set up data arrays
  MatrixXf estimates((lasso_init.n_boots_est/lasso_init.n_groups) * (lasso_init.n_lambdas/lasso_init.n_minigroups), ncols);
  MatrixXf scores((lasso_init.n_boots_est/lasso_init.n_groups), (lasso_init.n_lambdas/lasso_init.n_minigroups));
  MatrixXf best_estimates(lasso_init.n_boots_est, ncols);
  MatrixXf X_train_(train_split, ncols), X_test_(qrows-train_split, ncols);
  VectorXf y_train_(train_split), y_test_(qrows-train_split);
  //either we plan on using a test set, or we'll use the entire dataset for training
  if ( lasso_init.train_frac_overall < 1)
  {

    PermutationMatrix<Dynamic,Dynamic> perm_(qrows);
    perm_ = RandomPermute(qrows);
    X_est = perm_ * X_est;
    y_est = perm_ * y_est;
    X_train_ = X_est.topRows(train_split);
    y_train_ = y_est.head(train_split);
    X_test_ = X_est.bottomRows(qrows - train_split);
    y_test_ = y_est.tail(qrows - train_split);
  }
  else
  {
    X_train_ = X_est;
    y_train_ = y_est;
  }

  //Free memory
  X_est.resize(0,0);

  //containers for estimation
  VectorXf y_boot(boot_train_split), y_hat_boot(train_split-boot_train_split), y_true_boot(train_split-boot_train_split), y_test_boot(train_split-boot_train_split);
  MatrixXf X_boot(boot_train_split, ncols), X_boot_test(train_split-boot_train_split, ncols), X_recon;
  VectorXf z, r;
  vector<int> supportids;
  VectorXf z_est(ncols);

  double est_time_s, est_time_e, lasso_s, lasso_e, lasso_comm, lasso_est_comm, las_comm_2;

  if (rank == 0)
    est_time_s = MPI_Wtime();

  PermutationMatrix<Dynamic,Dynamic> perm_(X_train_.rows());

  //iterate over bootstrap samples
  for (int bootstrap=0; bootstrap < (lasso_init.n_boots_est/lasso_init.n_groups); bootstrap++)
  {
    perm_ = RandomPermute(X_train_.rows()); 
    X_train_ = perm_ * X_train_;
    y_train_ = perm_ * y_train_;
    y_boot = y_train_.head(boot_train_split);
    X_boot = X_train_.topRows(boot_train_split);
    X_boot_test = X_train_.bottomRows(train_split-boot_train_split);
    //extract the bootstrap indices, keeping a fraction of the data available for testing

    for (int lamb_idx=0; lamb_idx < (lasso_init.n_lambdas/lasso_init.n_minigroups); lamb_idx++)
    {
      z_est.setZero();
      lasso_comm = 0.0; 

      if (rank == 0)
        lasso_s = MPI_Wtime();

      //apply support : updated with ApplySupport routine. Concept: With the selected supports from Model Selection select only those X column (features).
      //With this updated X (Reconstructed X with small support) compute z. Inserted the obtained into the long z with complete feature list.

      tie(X_recon, supportids) = ApplySupport(X_boot, supports_.row(lamb_idx));


      if (supportids.size() != 0) {
      	tie(z,lasso_comm) =
        	lasso(X_recon, y_boot.array()-y_boot.mean(), 0, lasso_init.max_iter, lasso_init.reltol, lasso_init.abstol, lasso_init.rho, comm_mg);

      	for (int i = 0; i<supportids.size(); i++)
        	z_est(supportids[i]) = z(i);
      }

      if (rank == 0) {
        lasso_e += MPI_Wtime() - lasso_s;
        lasso_est_comm += lasso_comm;
      }

      if (rank_mg==0) {
        //store the fitted coefficients
        estimates.row((bootstrap*lasso_init.n_lambdas/lasso_init.n_minigroups)+lamb_idx) = z_est.unaryExpr(ptr_fun(not_NaN));
        y_hat_boot = X_boot_test * z_est;
        y_test_boot = y_train_.tail(train_split-boot_train_split);
        y_true_boot = y_test_boot.array()-y_test_boot.mean();
        //scores(bootstrap, lamb_idx) = r2_score(y_true, y_hat);
        //calculate sum of squared residuals
        float rss = (y_hat_boot.array() - y_true_boot.array()).array().square().sum();
        //calculate BIC as our scoring function
        scores(bootstrap, lamb_idx) = BIC(ncols, boot_train_split, rss);
     }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  //Free memory
  X_train_.resize(0,0);
  X_boot.resize(0,0);
  X_boot_test.resize(0,0);

  if ( (rank == 0) && lasso_init.verbose)
  {
    est_time_e = MPI_Wtime() - est_time_s;
    cout << "Total Estimation time: " << est_time_e << "(s)" << endl;
    cout << "Total Estimation lasso communication time: " << lasso_est_comm << "(s)" << endl;
    cout << "Total Estimatiom lasso computation time: " << lasso_e - lasso_est_comm << "(s)" << endl;
  }

  if ( (rank == 0) && lasso_init.debug)
    print_matrix( scores, "./debug/scores.txt");


  double bagg_s, bagg_e;

  if ( rank == 0 )
    bagg_s = MPI_Wtime();
  //Prepare data for bagging.     
  MatrixXf estimates_l1_root, estimates_l2_root, scores_l2_root, scores_l1_root, scores_mg;

  if (rank_g == 0)
    scores_l1_root.setZero(lasso_init.n_boots_est, lasso_init.n_lambdas);
  
  if(rank_g==0) estimates_l1_root.setZero(lasso_init.n_boots_est*lasso_init.n_lambdas, ncols);

  if (rank_mg == 0) {
    scores_l2_root.setZero(lasso_init.n_boots_est/lasso_init.n_groups, lasso_init.n_lambdas);
    estimates_l2_root.setZero(lasso_init.n_boots_est*lasso_init.n_lambdas/lasso_init.n_groups, ncols);
    //scores_mg.setZero(scores.rows(), scores.cols());

  }

  //MPI_Allreduce(scores.data(), scores_mg.data(), scores.rows() * scores.cols(), MPI_FLOAT, MPI_SUM, comm_mg); //we can do a reduce too.
  //MPI_Reduce(scores.data(), scores_mg.data(), scores.rows() * scores.cols(), MPI_FLOAT, MPI_SUM, 0, comm_mg);

  //if (rank_mg==0)
  //  scores_mg /= nprocs_mg;


  if (rank==0 && lasso_init.debug) cout << "Completed Reduce Est" << endl;

  if (MPI_COMM_NULL != L2_roots_comm) {
    MPI_Gather(scores.data(), scores.rows() * scores.cols(), MPI_FLOAT,
        scores_l2_root.data(), scores.rows() * scores.cols(), MPI_FLOAT, 0, L2_roots_comm);
    MPI_Gather(estimates.data(), estimates.rows() * estimates.cols() , MPI_FLOAT,
		estimates_l2_root.data(), estimates.rows() * estimates.cols(), MPI_FLOAT, 0, L2_roots_comm);
  }

   if (rank==0 && lasso_init.debug) cout << "Completed L2 Gather" << endl;

  if (MPI_COMM_NULL != L1_roots_comm)
  {
    MPI_Gather(scores_l2_root.data(), scores_l2_root.rows() * scores_l2_root.cols(), MPI_FLOAT,
        scores_l1_root.data(), scores_l2_root.rows() * scores_l2_root.cols(), MPI_FLOAT,0,  L1_roots_comm);

    MPI_Gather(estimates_l2_root.data(), estimates_l2_root.rows() * estimates_l2_root.cols(), MPI_FLOAT,
        estimates_l1_root.data(), estimates_l2_root.rows() * estimates_l2_root.cols(), MPI_FLOAT, 0, L1_roots_comm);
  }


  if (rank==0 && lasso_init.debug) cout << "Completed Gathers Est" << endl; 

  if (rank==0)
  {
    switch (lasso_init.bagging_options)
    {
      case 1:
        {
          //bagging option 1: for each bootstrap sample, find the regularization parameter that gave the best results
          for (int bootstrap=0; bootstrap < lasso_init.n_boots_est; bootstrap++)
          {
            VectorXf::Index lambda_max;
            float max = scores_l1_root.row(bootstrap).maxCoeff( &lambda_max);
            int lambda_max_idx_ = (int) lambda_max;
 	    if(bootstrap==0 && lasso_init.debug) cout << "passed this case 1: est_l1 size: " << estimates_l1_root.rows() << ", " << estimates_l1_root.cols() << endl;
            best_estimates.row(bootstrap) = estimates_l1_root.row((bootstrap*lasso_init.n_boots_est)+lambda_max_idx_);
          }
	  
	  //cout << "Passed case 1 loop" << endl;
          //take the median across estimates for the final, bagged estimate
          coef_ = median(best_estimates);
	  
  	  //cout << "passed median here" << endl;
          break;
        }
      case 2:
        {
          //bagging option 2: average estimates across bootstraps, and then find the regularization parameter that gives the best results
          VectorXf mean_scores;
          mean_scores = scores_l1_root.colwise().mean();
          VectorXf::Index lambda_max;
          float max = mean_scores.maxCoeff( &lambda_max);
          int lambda_max_idx_ = (int) lambda_max;

          for (int bootstrap=0; bootstrap < lasso_init.n_boots_est; bootstrap++)
            best_estimates.row(bootstrap) = estimates_l1_root.row(lambda_max_idx_);

          coef_ = median(best_estimates);
          break;
        }
      default:
        {
          cerr << "Bagging option " << lasso_init.bagging_options << " is not available.";
          break;
        }
    }
  }

  if (rank == 0)
  {
    bagg_e = MPI_Wtime() - bagg_s;
    cout << "Bagging time: " << bagg_e << "(s)" << endl;
  }
  if ( ( rank == 0 ) && (lasso_init.train_frac_overall < 1) )
  {
    //finally, see how the bagged estimates perform on the test set
    VectorXf y_hat_;
    y_hat_ = X_test_ * coef_;
    VectorXf y_true_;
    y_true_ = y_test_.array()-y_test_.mean();
    scores_ = pearson(y_hat_, y_true_);           
    //scores_ = r2_score(y_hat_, y_true_);

  }
  else
    scores_ = 0.0;

  if ( (rank == 0) && lasso_init.verbose) { cout << "Final score --> " << scores_ << endl; }

  if ( (rank == 0) && lasso_init.debug)
    print_vector( coef_, "./debug/coef_.txt");


  double w_time_s, w_time_e;

  if ( rank == 0 )
    w_time_s = MPI_Wtime();


  if ( MPI_COMM_NULL != L1_roots_comm) {
    MPI_Bcast(coef_.data(), ncols, MPI_FLOAT, 0, L1_roots_comm);
    MPI_Bcast(scores_l1_root.data(), scores_l1_root.rows() * scores_l1_root.cols(), MPI_FLOAT, 0, L1_roots_comm);

  }

  //write data into a hdf5 file
  //uses parallel write hdf5.

  if ( MPI_COMM_NULL != L1_roots_comm) {
    VectorXf _coef_(ncols);
    _coef_ = coef_;
    float *b_hat;
    float *bic_scores;
    b_hat  = (float*) malloc ( (ncols)/l1_root_size * sizeof(float) );
    bic_scores = (float* ) malloc (scores_l1_root.rows()/l1_root_size *  scores_l1_root.cols() * sizeof(float) );
    Map<VectorXf> (b_hat, (ncols)/l1_root_size) =
      _coef_.segment(l1_root_rank * (ncols)/l1_root_size, (ncols)/l1_root_size );
    Map<MatrixXf> (bic_scores, scores_l1_root.rows()/l1_root_size, scores_l1_root.cols()) =
      scores_l1_root.block(l1_root_rank * scores_l1_root.rows()/l1_root_size,
          0, scores_l1_root.rows()/l1_root_size, scores_l1_root.cols());

    write_out (ncols, 1, b_hat, lasso_init.Outfile1, L1_roots_comm, "coef_");
    write_out (lasso_init.n_boots_est, lasso_init.n_lambdas, bic_scores, lasso_init.Outfile2, L1_roots_comm, "scores_");
  }


  if ( (rank == 0) && lasso_init.verbose) {
    w_time_e = MPI_Wtime() - w_time_s;
    cout << "Save time: " << w_time_e << endl;
  }


  double time_max;
  toc = MPI_Wtime() - tic;
  MPI_Reduce(&toc, &time_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    cout << "Total time: " << time_max << endl;
    cout << "---> UoI Lasso complete." << endl;
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
