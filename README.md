# Distributed-UoI_Lasso 
			 Updated with BOOST CommandLine and rectified lasso_admm 
			tested with real dataset and validated with sklearn Lasso results
                    Runs tested on Cori NERSC supercomputer. 


## Requirements

languages: C, C++

computing resources: The model is created for `Cori KNL`. 

API and Libraries: MPI, HDF5-parallel, eigen3, gsl

`The version of MPICH is reverted back to 7.4.4 because of an open ticket regarding the new mpich.`

## Installation

1. Clone the module into your directory
2. `source load.sh`
3. `make`

## Usage

1. Edit the "job" script for input file path (vi job)
2. sbatch job (to submit the job)
3. Enjoy!!


## Working with Commandline arguments
1. In the job script you can see the srun command. After $exe you can add your commandline arguments.
2. The following are the commandlines available for UoI_Lasso


`[-l  | --n_lambdas]                  number of L1 penalty values to compare across (effectively sets the hyperparameter sweep)`

`[-s  | --selection_thres_frac]       used for soft thresholding in the selection step. normally, UoI-Lasso
                                      requires regressors to be selected in _all_ bootstraps to be selected
                                      for use in the estimation module. this requirement can be softened with
                                      this variable, by requiring that a regressor appear in
                                      selection_thres_frac of the bootstraps.`

` [--output_coef ]                     Output File containing final coefficients (coef_)`

` [--output_scores ]                   Output File containing final scores (scores_)`

` [--dataset_matrix ]                  String of data matrix name and structure in the hdf5 file`

` [--dataset_vector ]                  string of data vector  name and structure in the hdf5 file`

` [-t  | --train_frac_sel]             fraction of dataset to be used for training in the selection module. `

` [-T  | --train_frac_est]             fraction of dataset to be used for training in each bootstrap in the estimation module.`

` [-O  | --train_frac_overall]         fraction of dataset to be used for training in the overall estimation module.`

` [-c  | --n_boots_coarse]             number of bootstraps to use in the coarse lasso sweep.`

` [-b  | --n_boots_sel]                number of bootstraps to use in the selection module (dense lasso sweep).`

` [-e  | --n_boots_est]                number of bootstraps to use in the estimation module.`

` [-g  | --bagging_options]            equal to 1: for each bootstrap sample, find the regularization
                                             parameter that gave the best results
                                       equal to 2: average estimates across bootstraps, and then find the
                                             regularization parameter that gives the best results`

` [-a  | --use_admm ]                  flag indicating whether to use the ADMM algorithm.`

` [-f  | --file ] ARG (std::string)    Input File containing matrix and response vector`

` [-v  | --verbose ]                   verbose option`

` [-d  | --debug ]                     Debug option  boolean`

` [-n  | --n_groups ]                  Number of groups the or parallel bootstraps.`

` [--n_minigroups ]                    Number of parallel lambda executions.`

` [--n_coarse ]                        Number of parallel coarse bootstraps.`

` [--n_minicoarse ]                    Number of parallel coarse lambda.`

` [--max_iter ]                        Maximum number of iterations for Lasso ADMM`

` [--reltol ]                          RELTOL hyperparameter variable for Lasso ADMM`

` [--abstol ]                          ABSTOL hyperparameter variable for Lasso ADMM`

` [--rho ]                             rho variable for Lasso ADMM`



3. Mandatory arguments are `-f` or `--file` followed by the path of you file, `--dataset_matrix` followed by the hierarchical structure of your Matrix e.g. /data/X and `--dataset_vector` followed by the hierarchical structure of your vector e.g. /data/y.

4. If you do not mention other parameters, they are set to default. `--use_admm` is always ON because we use LASSO_ADMM right now. 

5. Example `job` script will have everything, just edit the necessary paths for your file.

6. If using NERSC, please store your hdf5 input file in $SCRATCH and edit the path in `job`.

7. By default your output b_hat coefficients will be stored in a file called coefs.h5 in the running directory. You can change this by `--output_coef` followed by the path and name of the file. 

8. Please note that unlike python the commandlines will not have a trailing = sign. For example `--dataset_matrix=/data/X` is invalid, whereas `--dataset_matrix /data/X` is valid.

## Contributing

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

## History

TODO: Write history

## Credits

Author: Mahesh Balasubramanian (guidance from Kris Bouchard, Prabhat, Brandon Cook)

Version: 2.0


## License

TODO: Write license

## Detailed description of the directory

1. `load.sh` : Has the required modules for the correct execution of UoI_Lasso application

