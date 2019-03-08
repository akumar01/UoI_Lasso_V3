#ifndef STRUCT_H
#define STRUCT_H

//Lasso input variable structure
struct INIT
{       
        int n_lambdas;
        float selection_thres_frac;
        float train_frac_sel;
        float train_frac_est;
        float train_frac_overall;
        int n_boots_coarse;
        int n_boots_sel;
        int n_boots_est;
        int bagging_options;
	int n_groups;
	int n_minigroups;
	int n_coarse;
	int n_minicoarse;
        bool use_admm;
        bool verbose;
        bool debug;
        char* Infile;
	char* Outfile1;
	char* Outfile2;
	char* data_mat;
	char* data_vec;
	int max_iter;
        float reltol;
        float abstol;
        float rho; 
};


/*struct OUT
{
	Eigen::VectorXf estimates;
	Eigen::VectorXf scores;

};*/

#endif //end STRUCT_H
