#include "utils.h"
#include "lasso.h"
#include "structure.h"
#include "manage-data.h"
#include "UoI_Lasso.h"
#include "bins.h"
#include "distribute-data.h"
//using namespace Eigen;

//########################
//Funcs used in UoI_Lasso
//########################
void print_matrix( MatrixXf , string );
void print_vector( VectorXf , string );
VectorXf logspace (int, int, int);
inline float BIC(float, float, float);
VectorXf median (MatrixXf); 
float r2_score (const VectorXf& , const VectorXf&);
//float pearson (VectorXf, VectorXf);
inline float r2_score (const VectorXf& , const VectorXf&);
PermutationMatrix<Dynamic,Dynamic> RandomPermute(int);
MatrixXf CreateSupport(int, int, int, MatrixXf);
tuple<MatrixXf, MatrixXf, double, double> lasso_sweep (MatrixXf, VectorXf, VectorXf, float, int, bool, bool, int, float, float, float, MPI_Comm);
tuple<MatrixXf,vector<int> > ApplySupport(MatrixXf&, VectorXf); 
void shuffle(int *, size_t); 

//########################
//Funcs definitions
//########################


VectorXf UoI_Lasso(INIT &init, MPI_Comm world_comm)
{
	//Initialize MPI for UoI_Lasso processes the processes
  	int rank, nprocs;
  	MPI_Comm_size(world_comm, &nprocs);
  	MPI_Comm_rank(world_comm, &rank);
	MPI_Group world_group;
   	MPI_Comm_group(world_comm, &world_group);

	int n_samples_;
	int n_features_;
	double start, end;
	
	if (rank == 0)
		start = MPI_Wtime(); 

	if (rank == 0)
	{
		 //extract model dimensions from design matrix
                n_samples_= get_rows(init.Infile, init.data_mat);
                n_features_ = get_cols(init.Infile, init.data_mat);
	}

	MPI_Bcast(&n_samples_, 1, MPI_INT, 0, world_comm);		
	MPI_Bcast(&n_features_, 1, MPI_INT, 0, world_comm);

	if ( (rank == 0 ) && init.verbose) {
		cout << "nprocs: " << nprocs << endl;
		cout << "n_samples: " << n_samples_ << endl;
		cout << "n_features: " << n_features_ << endl;
	}
	//check for processors limits
	if ( (rank == 0 ) && (nprocs > n_samples_) ) 
	{
      		printf("must have nprocs < n_samples_ \n");
      		fflush(stdout);
      		MPI_Abort(world_comm, 3);
    	}

    	if ( ( rank == 0) && ( init.n_groups > nprocs ) ) 
	{
      		printf("must have ngroups < nprocs \n");
      		fflush(stdout);
      		MPI_Abort(world_comm, 4);
    	}

	double load_time_s, load_time_e;


	if (rank == 0) 
		load_time_s = MPI_Wtime();

	int local_rows = bin_size_1D(rank, n_samples_, nprocs);
	//MatrixXf X;
	Map<Matrix<float, Dynamic, Dynamic, RowMajor>> X( get_matrix(local_rows, n_features_, n_samples_, world_comm, rank, init.data_mat, init.Infile), local_rows, n_features_);
	//VectorXf y;
	Map<VectorXf> y( get_array(local_rows, n_samples_, world_comm, rank, init.data_vec, init.Infile), local_rows);
	
	if ( (rank == 0 ) && init.verbose) 
	{	
		load_time_e = MPI_Wtime() - load_time_s;
		cout << "(1) Loaded data.\n" << n_samples_ << " samples with " << n_features_ << " features."	<< endl;
		cout << "Loading time: " << load_time_e << "(s)" << endl;

	}

	if ( ( rank == 0 ) && init.debug)	
	{
		print_matrix( X, "./debug/X_1.txt" );
		print_vector( y, "./debug/y_1.txt" );
	}

	//combine the matrix and the vector into 1 matrix for random distribution
	MatrixXf A_(local_rows, (n_features_+1));
	A_ << X,y;
	float *A;
   	size_t sizeA = (size_t) A_.rows() * (size_t) A_.cols() * sizeof(float);
	A = (float*) malloc (sizeA);
	Map<MatrixXf> (A, A_.rows(), A_.cols() ) = A_;

	//cout << "A local for rank: " << rank << " sizeA: " <<  sizeA  << "  size: " << sizeA_  << endl;

  	if ( (rank == 0) && init.debug)
		print_matrix( A_, "./debug/A.txt");
	
	if ( (rank ==0 ) && init.verbose )
		cout << "Preparing data for random distribution..." << endl;


	//create color with init.p_coarse for level1 and level2 parallelization for coarse
	int color = bin_coord_1D(rank, nprocs, init.n_coarse);
        MPI_Comm comm_c;
        MPI_Comm_split(world_comm, color, rank, &comm_c);

        int nprocs_c, rank_c;
        MPI_Comm_size(comm_c, &nprocs_c);
        MPI_Comm_rank(comm_c, &rank_c);
        MPI_Group n_coarse_group;
        MPI_Comm_group(comm_c, &n_coarse_group);

        color = bin_coord_1D(rank_c, nprocs_c, init.n_minicoarse);
        MPI_Comm comm_mc;
        MPI_Comm_split(comm_c, color, rank_c, &comm_mc);

        int nprocs_mc, rank_mc;
        MPI_Comm_size(comm_mc, &nprocs_mc);
        MPI_Comm_rank(comm_mc, &rank_mc);

	//Level-1 root ranks group creation.
        int cboot_root_rank = -1, cboot_root_size = -1;

        VectorXi cboot_roots(init.n_coarse);
        int root = 0;

        for (int i=0; i<init.n_coarse; i++)
        {
                root += i * nprocs_c;
                cboot_roots(i) = root;
        }

        MPI_Group cboot_root_group;
        MPI_Group_incl(world_group, init.n_coarse, cboot_roots.data(), &cboot_root_group);
        MPI_Comm cboot_roots_comm;
        MPI_Comm_create_group(world_comm, cboot_root_group, 0, &cboot_roots_comm);

        if (MPI_COMM_NULL != cboot_roots_comm)
        {
                MPI_Comm_rank(cboot_roots_comm, &cboot_root_rank);
                MPI_Comm_size(cboot_roots_comm, &cboot_root_size);
        }

        //Level-2 root ranks group creation.
        int clam_root_rank = -1, clam_root_size = -1;

        VectorXi clam_roots(init.n_minicoarse);
        root = 0;
	
	for (int i=0; i<init.n_minicoarse; i++)
        {
                root += i * nprocs_mc;
                clam_roots(i) = root;
        }

        MPI_Group clam_root_group;
        MPI_Group_incl(cboot_root_group, init.n_minicoarse, clam_roots.data(), &clam_root_group);
        MPI_Comm clam_roots_comm;
        MPI_Comm_create_group(comm_c, clam_root_group, 0, &clam_roots_comm);

        if (MPI_COMM_NULL != clam_roots_comm)
        {
                MPI_Comm_rank(clam_roots_comm, &clam_root_rank);
                MPI_Comm_size(clam_roots_comm, &clam_root_size);
        }

	//------------------------------------------------
	//create color with init.n_groups for level 1 parallelization
  	color = bin_coord_1D(rank, nprocs, init.n_groups);
  	MPI_Comm comm_g;
  	MPI_Comm_split(world_comm, color, rank, &comm_g);
 
 	int nprocs_g, rank_g;
  	MPI_Comm_size(comm_g, &nprocs_g);
  	MPI_Comm_rank(comm_g, &rank_g);
	MPI_Group L1_group;
    	MPI_Comm_group(comm_g, &L1_group);

	//create color iwth init.n_minigroups for level 2 parallelism
	color = bin_coord_1D(rank_g, nprocs_g, init.n_minigroups);
	MPI_Comm comm_mg;
	MPI_Comm_split(comm_g, color, rank_g, &comm_mg);

	int nprocs_mg, rank_mg;
	MPI_Comm_size(comm_mg, &nprocs_mg);
	MPI_Comm_rank(comm_mg, &rank_mg);

	//create L1 and L2 root processes as separate groups.
	//This is for easier output management.

	//Level-1 root ranks group creation.
	int l1_root_rank = -1, l1_root_size = -1;

	VectorXi l1_roots(init.n_groups);
        root = 0;

        for (int i=0; i<init.n_groups; i++)
        {
                root += i * nprocs_g;
                l1_roots(i) = root;
        }

        MPI_Group l1_root_group;
        MPI_Group_incl(world_group, init.n_groups, l1_roots.data(), &l1_root_group);
        MPI_Comm L1_roots_comm;
        MPI_Comm_create_group(world_comm, l1_root_group, 0, &L1_roots_comm);  

	if (MPI_COMM_NULL != L1_roots_comm) 
	{
         	MPI_Comm_rank(L1_roots_comm, &l1_root_rank);
         	MPI_Comm_size(L1_roots_comm, &l1_root_size);
    	}

	//Level-2 root ranks group creation.
	int l2_root_rank = -1, l2_root_size = -1;

	VectorXi l2_roots(init.n_minigroups);
    	root = 0;

    	for (int i=0; i<init.n_minigroups; i++) 
	{
        	root += i * nprocs_mg;
		l2_roots(i) = root;
    	}

   	MPI_Group l2_root_group;
    	MPI_Group_incl(L1_group, init.n_minigroups, l2_roots.data(), &l2_root_group);
    	MPI_Comm L2_roots_comm;
    	MPI_Comm_create_group(comm_g, l2_root_group, 0, &L2_roots_comm); 

	if (MPI_COMM_NULL != L2_roots_comm)
        {
                MPI_Comm_rank(L2_roots_comm, &l2_root_rank);
                MPI_Comm_size(L2_roots_comm, &l2_root_size);
        }

	//All the parameters are set. 
	//--------------------------------------------------------------

  	int qrows = bin_size_1D(rank_c, n_samples_, nprocs_c);
	//int qrows = floor(n_samples_/nprocs_c);
  	int qcols = n_features_ + 1;

	float *B_;
	size_t sizeB = (size_t) qrows * (size_t) qcols * sizeof(float); 
  	B_ = (float *) malloc( sizeB );

	double dis_time_s, dis_time_e;
	
	if (rank == 0)
		dis_time_s = MPI_Wtime();


 	distribute_data (A,  local_rows, qrows, n_samples_, qcols, n_samples_, B_, world_comm, comm_c);	

	if ( (rank == 0 ) && init.verbose ) 
	{
		dis_time_e = MPI_Wtime() - dis_time_s;
                cout << "Random distribution done." << endl;
		cout << "Random Distribution time: " << dis_time_e << "(s)" << endl;
		MPI_Abort(world_comm, 911);
	}
	
	MPI_Barrier( world_comm );

	Map<MatrixXf> B( B_, qrows, qcols);	

	if ( (rank == 69) && init.debug)
                print_matrix( B, "./debug/B_69.txt");


	MatrixXf X_coarse(X.rows(), X.cols()); 
	X_coarse = B.block(0, 0, B.rows(), B.cols()-1);
	VectorXf y_coarse(y.size());
	y_coarse = B.rightCols(1); 
		
	if ( (rank == 0) && init.debug)
	{
		print_matrix(B, "./debug/B.txt");
                print_matrix( X_coarse, "./debug/X_c.txt");
		print_vector( y_coarse, "./debug/y_c.txt");
	}

	//free(B_);
	//perform an initial coarse sweep over the lambda parameters
        //this is to zero-in on the relevant regularization region.
	
	VectorXf lambda_coarse(init.n_lambdas);
	VectorXf lambda_coarse_dis(init.n_lambdas/init.n_minicoarse);

	if (rank_g == 0 )
	{
		if (init.n_lambdas == 1)
			lambda_coarse.setOnes();
		else
			lambda_coarse = logspace(-3, 3, init.n_lambdas); 
	}	

	MPI_Bcast(lambda_coarse.data(), init.n_lambdas, MPI_FLOAT, 0, world_comm);

        if (MPI_COMM_NULL != clam_roots_comm)
                lambda_coarse_dis = lambda_coarse.segment(clam_root_rank*(init.n_lambdas/init.n_minicoarse), init.n_lambdas/init.n_minicoarse);

	if (rank == 0)
		cout << "Passed first coarse " << endl;


        //if (MPI_COMM_NULL != clam_roots_comm)
        MPI_Bcast(lambda_coarse_dis.data(), lambda_coarse_dis.size(), MPI_FLOAT, 0, comm_mc);

	if (rank == 0)
               cout << "Passed second coarse " << endl;

	if ( rank == 0 )
		cout << "lambda coarse created for " << lambda_coarse.size() << " size." << endl;
	
	MatrixXf estimates_coarse, scores_coarse;
	double time1, time2;
	double coarse_lass, coarse_lass_comm; 
	//run the coarse lasso sweep	
	
	if (rank == 0)
		time1 = MPI_Wtime();

	tie (estimates_coarse, scores_coarse, coarse_lass, coarse_lass_comm)  = 
					lasso_sweep (X_coarse, y_coarse, lambda_coarse, init.train_frac_sel, init.n_boots_coarse/init.n_coarse,
									init.use_admm, init.debug, init.max_iter, init.reltol, init.abstol,
                                                                                init.rho, comm_mc);

	//---------------------------------------------------------------
	//Collect data 
	MatrixXf estimates_cboot_root, scores_cboot_root, scores_clam_root, scores_mc(scores_coarse.rows(), scores_coarse.cols());

        if (rank_c == 0)
        {
                MatrixXf estimates_cboot_root(init.n_boots_est, init.n_lambdas);
                MatrixXf score_cboot_root(init.n_boots_est, init.n_lambdas);
        }

        if (rank_mc == 0)
                MatrixXf scores_clam_root(init.n_coarse/init.n_coarse, init.n_lambdas);

        MPI_Allreduce(scores_coarse.data(), scores_mc.data(), scores_coarse.rows() * scores_coarse.cols(), MPI_FLOAT, MPI_SUM, comm_mc); //we can do a reduce too.
        scores_mc /= nprocs_mc;

        if (MPI_COMM_NULL != clam_roots_comm)
                MPI_Gather(scores_mc.data(), scores_mc.rows() * scores_mc.cols(), MPI_FLOAT,
                                scores_clam_root.data(), scores_mc.rows() * scores_mc.cols(), MPI_FLOAT, 0, clam_roots_comm);

        if (MPI_COMM_NULL != cboot_roots_comm)
        {
                MPI_Allgather(scores_clam_root.data(), scores_clam_root.rows() * scores_clam_root.cols(), MPI_FLOAT,
                                scores_cboot_root.data(), scores_clam_root.rows() * scores_clam_root.cols(), MPI_FLOAT, cboot_roots_comm);

                MPI_Allgather(estimates_coarse.data(), estimates_cboot_root.rows() * estimates_cboot_root.cols(), MPI_FLOAT,
                                estimates_cboot_root.data(), estimates_cboot_root.rows() * estimates_cboot_root.cols(), MPI_FLOAT, cboot_roots_comm);
        }

	//-------------------------------------------------------------

	if ( (rank == 0) && init.debug)
        {
                print_matrix( estimates_coarse, "./debug/estimated_coarse.txt");
                print_matrix( scores_coarse, "./debug/scores_coarse.txt");
        }

	//deduce the index which maximizes the explained variance over bootstraps
	float d_lambda, lambda_max;
	if ( rank == 0 )
	{
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
	
	if ( (rank == 0) && init.debug)
        {
                //print_vector( mean, "./debug/mean_vector.txt");
                print_vector( lambda_coarse, "./debug/lambda_coarse.txt");
		cout << "lambda_max: " << lambda_max << endl;

        }

	 //now that we've narrowed down the regularization parameters,
         //we'll run a dense sweep which begins the model selection module of UoI
	if ((rank == 0) && init.verbose)
	{
		time2 = MPI_Wtime() - time1;
		cout << "Coarse sweep time: " << time2 << "(s)" << endl;
	}
	//#######################
        //### Model Selection ###
        // #######################
	
	double sel_time_s, sel_time_e;

	if ( (rank == 0) && init.verbose)
		cout << "(2) Beginning model selection. Exploring penalty region centered at " << lambda_max << "." << endl;

	VectorXf lambdas(init.n_lambdas);
	VectorXf lambda_dis(init.n_lambdas/init.n_minigroups );

        if (rank == 0 )
        {
                if (init.n_lambdas == 1)
                        lambdas << lambda_max;
                else
                        lambdas.setLinSpaced(init.n_lambdas, lambda_max - 5 * d_lambda, lambda_max + 5 * d_lambda);
        }	
		
	MPI_Bcast(lambdas.data(), init.n_lambdas, MPI_FLOAT, 0, world_comm);


	if ( rank_mg == 0 )
                lambda_dis = lambdas.segment(l2_root_rank*(init.n_lambdas/init.n_minigroups), init.n_lambdas/init.n_minigroups);

  
	if (MPI_COMM_NULL != L2_roots_comm) 
        	MPI_Bcast(lambda_dis.data(), lambda_dis.size(), MPI_FLOAT, 0, comm_mg);

	qrows = bin_size_1D(rank_mg, n_samples_, nprocs_mg);
	//qrows = floor(n_samples_/nprocs_mg);
	float *B__;
        B__ = (float *) malloc( qrows * qcols * sizeof(float) );


	double par_dis_time_s, par_dis_time_e;

        if (rank == 0)
                dis_time_s = MPI_Wtime();

        //distribute_data (A,  local_rows, qrows, n_samples_, n_features_, n_samples_, B__, world_comm, comm_mg);
	
	if ( (rank == 0 ) && init.verbose )
        {
                par_dis_time_e = MPI_Wtime() - par_dis_time_s;
                cout << "Random distribution for selection done." << endl;
                cout << "Random Distribution for selection time: " << dis_time_e << "(s)" << endl;
        }

	Map<MatrixXf> B_sel( B__, qrows, qcols);
	MatrixXf X_(B_sel.rows(), B_sel.cols()-1);
        X_ = B_sel.block(0, 0, B_sel.rows(), B_sel.cols()-1);
        VectorXf y_(B_sel.rows());
        y_ = B_sel.rightCols(1);
	double lass_time, lass_comm;

	MatrixXf estimates_dense, scores_dense;
	if (rank == 0)
		sel_time_s = MPI_Wtime();
	
	tie (estimates_dense, scores_dense,lass_time, lass_comm)  = 
				lasso_sweep (X_, y_, lambda_dis, init.train_frac_sel, (init.n_boots_sel/init.n_groups),
                                                                        init.use_admm, init.debug, init.max_iter, init.reltol, init.abstol, 
										init.rho, comm_mg);

	double sup_time_s, sup_time_e;

	if (rank == 0) 
	{
		sel_time_e = MPI_Wtime() - sel_time_s;
		sup_time_s = MPI_Wtime();
	}

	if ( rank == 0 &&  init.verbose )
        {
                cout << "Total Selection lasso time: " << sel_time_e << "(s)" << endl;
                cout << "Total Lasso communication time: " << lass_comm << "(s)" << endl;
                cout << "Total Lasso computation time: " << lass_time-lass_comm << "(s)" << endl;
        }

	if ( (rank == 0) && init.debug)
        {
                print_matrix( estimates_dense, "./debug/estimated_dense.txt");
                print_matrix( scores_dense, "./debug/scores_dense.txt");
        }

	//intersect supports across bootstraps for each lambda value
        //we impose a (potentially) soft intersection

	int threshold = (init.selection_thres_frac * (init.n_boots_sel/init.n_groups));
	
	//create support matrix storage
	MatrixXf supports_((init.n_lambdas/init.n_minigroups), n_features_);
	
	if ( rank_mg == 0)
		supports_ = CreateSupport((init.n_lambdas/init.n_minigroups), (init.n_boots_sel/init.n_groups), threshold, estimates_dense); 

	if (rank == 0)
		sup_time_e = MPI_Wtime() - sup_time_s;

	if ( (rank == 0) && init.debug)
                print_matrix( supports_, "./debug/supports_.txt");
	
	if ( MPI_COMM_NULL != L2_roots_comm )
		MPI_Bcast(supports_.data(), (init.n_lambdas/init.n_minigroups)*n_features_, MPI_FLOAT, 0, comm_mg);

	//#######################
        //### Model Estimation ###
        // #######################

	//we'll use the supports obtained in the selection module to calculate
        //bagged OLS estimates over bootstraps

	if ( (rank == 0 ) && init.verbose )
	{
                        cout << "(3) Model selection complete. " << endl;
			cout << "Model Selection time: " << sel_time_e << "(s)" << endl;
			cout << "Support creation time: " << sup_time_e << "(s)"<< endl;
			cout << " (4) Beginning model estimation, with " << init.n_boots_est << " bootstraps" << endl; 
	}
	//create or overwrite arrays to collect final results
	VectorXf   coef_   =  VectorXf::Zero(n_features_);
	float scores_; 
	//determine how many samples will be used for overall training
	int train_split = round( init.train_frac_overall * qrows);
	//determine how many samples will be used for training within a bootstrap	
	int boot_train_split = round( init.train_frac_est * train_split);

	//set up data arrays
	MatrixXf estimates((init.n_boots_est/init.n_groups) * (init.n_lambdas/init.n_minigroups), n_features_);
	MatrixXf scores((init.n_boots_est/init.n_groups), (init.n_lambdas/init.n_minigroups));
	MatrixXf best_estimates(init.n_boots_est, n_features_);	
	MatrixXf X_train_, X_test_;
	VectorXf y_train_, y_test_;	 
	//either we plan on using a test set, or we'll use the entire dataset for training
	if ( init.train_frac_overall < 1)
	{
			
		PermutationMatrix<Dynamic,Dynamic> perm_(qrows);
                perm_ = RandomPermute(qrows);
                X_ = perm_ * X_;
                y_ = perm_ * y_;
		X_train_ = X_.topRows(train_split);
		y_train_ = y_.head(train_split);
		X_test_ = X_.bottomRows(qrows - train_split);
		y_test_ = y_.tail(qrows - train_split);
	}	
	else
	{	
		X_train_ = X_;
		y_train_ = y_;
	}
	
	//containers for estimation
	VectorXf y_boot, y_hat_boot, y_true_boot, y_test_boot;
        MatrixXf X_boot, X_boot_test;
	VectorXf z, r;
		
	double est_time_s, est_time_e, lasso_s, lasso_e, lasso_comm, lasso_est_comm; 
	
	if (rank == 0)
		est_time_s = MPI_Wtime();
	
	//iterate over bootstrap samples
	for (int bootstrap=0; bootstrap < (init.n_boots_est/init.n_groups); bootstrap++)
	{	
		PermutationMatrix<Dynamic,Dynamic> perm_(X_train_.rows());
                perm_ = RandomPermute(X_train_.rows());
                X_train_ = perm_ * X_train_;
                y_train_ = perm_ * y_train_;
		y_boot = y_train_.head(boot_train_split);
		X_boot = X_train_.topRows(boot_train_split);
		X_boot_test = X_train_.bottomRows(train_split-boot_train_split);
		//extract the bootstrap indices, keeping a fraction of the data available for testing
		vector<int> supportids;		
		VectorXf z_est(n_features_);

		for (int lamb_idx=0; lamb_idx < (init.n_lambdas/init.n_minigroups); lamb_idx++)
		{
			if (rank == 0)
				lasso_s = MPI_Wtime();

			MatrixXf X_recon;	
                        tie(X_recon, supportids) = ApplySupport(X_boot, supports_.row(lamb_idx));

                        tie(z,lasso_comm) = 
				lasso(X_recon, y_boot.array()-y_boot.mean(), 0, init.max_iter, init.reltol, init.abstol, init.rho, comm_mg);

			if (rank == 0)
			{
				lasso_e = MPI_Wtime() - lasso_s;
				lasso_est_comm += lasso_comm;
			}

			//apply support : can be changed if necessary. multiplies elementwise with the supports_ array to select only the supports.
			
			for (int i = 0; i<supportids.size(); i++)
				z_est(supportids[i]) = z(i);		

			//store the fitted coefficients
			estimates.row((bootstrap*init.n_lambdas)+lamb_idx) = z_est;
                        y_hat_boot = X_boot_test * z_est;
                        y_test_boot = y_train_.tail(train_split-boot_train_split);
                        y_true_boot = y_test_boot.array()-y_test_boot.mean();
                        //scores(bootstrap, lamb_idx) = r2_score(y_true, y_hat);
			//calculate sum of squared residuals
			float rss = (y_hat_boot.array() - y_true_boot.array()).square().sum();
			//calculate BIC as our scoring function
			scores(bootstrap, lamb_idx) = BIC(n_features_, boot_train_split, rss);
		}
	}

	if ( (rank == 0) && init.verbose)
	{
		est_time_e = MPI_Wtime() - est_time_s;
                cout << "Total Estimation time: " << est_time_e << "(s)" << endl;
                cout << "Total Estimation lasso communication time: " << lasso_est_comm << "(s)" << endl;
                cout << "Total Estimatiom lasso computation time: " << lasso_e - lasso_est_comm << "(s)" << endl;
	}

	 if ( (rank == 0) && init.debug)
                print_matrix( scores, "./debug/scores.txt");


	double bagg_s, bagg_e;
	
	if ( rank == 0 )
		bagg_s = MPI_Wtime(); 
	//Prepare data for bagging.	
	MatrixXf estimates_l1_root, scores_l2_root, scores_l1_root, scores_mg(scores.rows(), scores.cols());
	
	if (rank_g == 0)
	{
		MatrixXf estimates_l1_root(init.n_boots_est, init.n_lambdas);
		MatrixXf score_l1_root(init.n_boots_est, init.n_lambdas);
	}
	
	if (rank_mg == 0)
		MatrixXf scores_l2_root(init.n_boots_est/init.n_groups, init.n_lambdas);
 
	MPI_Allreduce(scores.data(), scores_mg.data(), scores.rows() * scores.cols(), MPI_FLOAT, MPI_SUM, comm_mg); //we can do a reduce too.
	scores_mg /= nprocs_mg;

	if (MPI_COMM_NULL != L2_roots_comm)
		MPI_Gather(scores_mg.data(), scores_mg.rows() * scores_mg.cols(), MPI_FLOAT, 
				scores_l2_root.data(), scores_mg.rows() * scores_mg.cols(), MPI_FLOAT, 0, L2_roots_comm);
	
	if (MPI_COMM_NULL != L1_roots_comm)
	{
		MPI_Allgather(scores_l2_root.data(), scores_l2_root.rows() * scores_l2_root.cols(), MPI_FLOAT, 
				scores_l1_root.data(), scores_l2_root.rows() * scores_l2_root.cols(), MPI_FLOAT, L1_roots_comm);
	
		MPI_Allgather(estimates.data(), estimates_l1_root.rows() * estimates_l1_root.cols(), MPI_FLOAT,
                                estimates_l1_root.data(), estimates_l1_root.rows() * estimates_l1_root.cols(), MPI_FLOAT, L1_roots_comm);
	}
	
	if (rank_g == 0)
	{
		switch (init.bagging_options) 
		{
			case 1:
			{
				//bagging option 1: for each bootstrap sample, find the regularization parameter that gave the best results
				for (int bootstrap=0; bootstrap < init.n_boots_est; bootstrap++)
				{
					VectorXf::Index lambda_max;
        				float max = scores_l1_root.row(bootstrap).maxCoeff( &lambda_max);
					int lambda_max_idx_ = (int) lambda_max; 
					best_estimates.row(bootstrap) = estimates_l1_root.row((bootstrap*init.n_boots_est)+lambda_max_idx_); 
				}	
		
				//take the median across estimates for the final, bagged estimate
				coef_ = median(best_estimates); 	
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

				for (int bootstrap=0; bootstrap < init.n_boots_est; bootstrap++)
					best_estimates.row(bootstrap) = estimates_l1_root.row(lambda_max_idx_);

				coef_ = median(best_estimates); 
				break;
			}
			default:
			{
				cerr << "Bagging option " << init.bagging_options << " is not available."; 
				break;
			}
		}
	}
	
	if (rank == 0)
	{
		bagg_e = MPI_Wtime() - bagg_s;
		cout << "Bagging time: " << bagg_e << "(s)" << endl;
	}
	if ( ( rank == 0 ) && (init.train_frac_overall < 1) )
	{
		//finally, see how the bagged estimates perform on the test set
		VectorXf y_hat_;
                y_hat_ = X_test_ * coef_;
                VectorXf y_true_;
                y_true_ = y_test_.array()-y_test_.mean();
                //scores_ = pearson(y_hat_, y_true_);		
		scores_ = r2_score(y_hat_, y_true_);
	
	}		
	else
		scores_ = 0.0;

	if ( (rank == 0) && init.verbose) { cout << "Final score --> " << scores_ << endl; }		

	if ( (rank == 0) && init.debug)
                print_vector( coef_, "./debug/coef_.txt");


	double w_time_s, w_time_e;

	if ( rank == 0 )	
		w_time_s = MPI_Wtime();

	//write data into a hdf5 file
	if ( MPI_COMM_NULL != L1_roots_comm)
	{
		VectorXf _coef_(n_features_);
		_coef_ = coef_;
		float *b_hat;
		float *bic_scores;
		b_hat  = (float*) malloc ( (n_features_)/l1_root_size * sizeof(float) );
		bic_scores = (float* ) malloc (scores_l1_root.rows()/l1_root_size *  scores_l1_root.cols() * sizeof(float) );
		Map<VectorXf> (b_hat, (n_features_)/l1_root_size) = 
				_coef_.segment(l1_root_rank * (n_features_)/l1_root_size, (n_features_)/l1_root_size );
		Map<MatrixXf> (bic_scores, scores_l1_root.rows()/l1_root_size, scores_l1_root.cols()) = 
		 			scores_l1_root.block(l1_root_rank * scores_l1_root.rows()/l1_root_size, 
								0, scores_l1_root.rows()/l1_root_size, scores_l1_root.cols());

		write_out (n_features_, 1, b_hat, init.Outfile1, L1_roots_comm, "coef_");	
		write_out (init.n_boots_est, init.n_lambdas, bic_scores, init.Outfile2, L1_roots_comm, "scores_");	
	}

	if ( (rank == 0) && init.verbose )
	{
		w_time_e = MPI_Wtime() - w_time_s;
		cout << "Save time: " << w_time_e << endl;
		end = MPI_Wtime() - start;
		cout << "Total UoI_Lasso time: " << end << endl; 
		cout << "---> UoI Lasso complete." << endl;
	}
	return coef_; 

}

VectorXf logspace (int start, int end, int size) {

    VectorXf vec;
    vec.setLinSpaced(size, start, end);

    for(int i=0; i<size; i++)
        vec(i) = pow(10,vec(i));

    return vec;
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
	MatrixXf X_perm(n_samples, n_features);
        VectorXf y_perm(n_samples);
	VectorXf y_train(n_train_samples);
	MatrixXf X_train(n_train_samples, n_features);
	VectorXf est_(n_features);
	VectorXf y_hat(n_samples-n_train_samples);
	VectorXf y_true(n_samples-n_train_samples);
        VectorXf y_test(n_samples-n_train_samples);
	MatrixXf X_test(n_samples-n_train_samples, n_features);

	double time1, time2, time3, time4;

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
		 	print_matrix( X_train, "./debug/X_train.txt");	
			print_matrix( X_test, "./debug/X_test.txt");
			print_vector( y_train, "./debug/y_train.txt");
			print_vector( y_test, "./debug/y_test.txt");

		}
		
		for (int lambda_idx=0; lambda_idx < n_lambdas; lambda_idx++)
		{
			//if (rank_sweep==0)
			//	cout << "top lambda_idx: " << lambda_idx << endl;
			float n_lamb = lambda(lambda_idx);
			 if ( rank_sweep == 0 )
                                time1 = MPI_Wtime();

			tie(est_,time3) = lasso(X_train, y_train.array()-y_train.mean(), n_lamb, MAX_ITER, RELTOL, ABSTOL, rho, comm_sweep);

			if(rank_sweep==0)
                        {
                                time2 += MPI_Wtime() - time1;
                                time4 += time3;
				cout << "time for 1 lasso: " << time2 << "(s)" << endl;
				cout << "time for 1 lasso comm: " << time4 << "(s)" << endl;
				cout << "time for 1 lasso comp: " << time2-time4 << "(s)" << endl;
				print_vector( est_, "./debug/est_dense_0_0.txt");
				MPI_Abort(comm_sweep, 23);
                        }
			//estimates.row((bootstrap*n_lambdas)+lambda_idx) = est_; this stores estimates from 0-n_lambdas consecutively 
			estimates.row((lambda_idx*n_bootstraps) + bootstrap) = est_; // this stores estimates in order 0 in n_lambda steps. so bootstraps are strored consecutively.
			y_hat = X_test * est_;
			y_true = y_test.array()-y_test.mean();
			//scores(bootstrap, lambda_idx) = pearson(y_hat, y_true);
			scores(bootstrap, lambda_idx) = r2_score(y_hat, y_true);

			if (rank_sweep==0)
			{
				cout << "Score --> " << scores(bootstrap, lambda_idx) << endl;
				MPI_Abort(comm_sweep, 23);
			}
		
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

inline float r2_score (const VectorXf& x, const VectorXf& y)
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
	// n_lambdas 	: int number of lambda parameters
	// n_bootstraps	: int number of sel bootstraps. 
	// threshold_	: int used for soft thresholding
	//estimates 	: (n_lambda) x (n_bootstraps) x (n_features) 
	
	//Output:
	//------------------------------------------
	// support	: (n_lambda) x (n_features) support 
	//TODO: support matrix is currently floa. Must check compatability and convert it into bool.



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

        float bic;
        return (bic = -n_samples * log(rss/n_samples) - n_features * log(n_samples));


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

/*float pearson (VectorXf vec1, VectorXf vec2) {


        VectorXd vec1_d = vec1.cast <double>();
        VectorXd vec2_d = vec2.cast <double>();

        gsl_vector_view gsl_x = gsl_vector_view_array( vec1_d.data(), vec1_d.size());
        gsl_vector_view gsl_y = gsl_vector_view_array( vec2_d.data(), vec2_d.size());

        gsl_vector *gsl_v1 =  &gsl_x.vector;
        gsl_vector *gsl_v2 = &gsl_y.vector;
        double r = gsl_stats_correlation (gsl_v1->data, 1, gsl_v2->data, 1, gsl_v1->size);
	
	float r2 = (float) pow(r, 2);	
	
        return r2;
}*/

int binary_to_decimal(const VectorXf& bits)
{
    int result = 0;
    int base = 1;

    //Supposing the MSB is at the begin of the bits vector:
    for(unsigned int i = bits.size()-1 ; i >= 0 ; --i)
    {
        result += (int) bits(i)*base;
        base *= 2;
    }

    return result;
}

vector<int> binary_search(const vector<int>& input, int id)
{
        vector<int> ret;

        for (int i =0;i<input.size(); i++)
                if (input[i] == id)
                        ret.push_back(i);

        return ret;

}


MatrixXf ReconA(MatrixXf& A, const vector<int>& id)
{
        MatrixXf ret(id.size(), A.cols());

        for (int i=0; i<id.size();i++)
                ret.row(i) = A.row(id[i]);

        return ret;


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
