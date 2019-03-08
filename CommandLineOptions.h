#ifndef _COMMAND_LINE_OPTONS_H_
#define _COMMAND_LINE_OPTONS_H_

#include <string>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

//----------------------------------------------------------
// Handle command line options
//
// Options: 
// [-h 	| --help] 			Show help menu
// [-V 	| --version]			Display version information
// [-l	| --n_lambdas]			number of L1 penalty values to compare across (effectively sets the hyperparameter sweep)
// [-s	| --selection_thres_frac] 	used for soft thresholding in the selection step. normally, UoI-Lasso
//			                requires regressors to be selected in _all_ bootstraps to be selected
// 				        for use in the estimation module. this requirement can be softened with
// 			             	this variable, by requiring that a regressor appear in
// 			                selection_thres_frac of the bootstraps.
// [--output_coef ]          		Output File containing final coefficients (coef_)
// [--output_scores ]      		Output File containing final scores (scores_)
// [--dataset_matrix ]			String of data matrix name and structure in the hdf5 file
// [--dataset_vector ]			string of data vector  name and structure in the hdf5 file
// [-t	| --train_frac_sel]		fraction of dataset to be used for training in the selection module. 
// [-T	| --train_frac_est]		fraction of dataset to be used for training in each bootstrap in the estimation module.
// [-O	| --train_frac_overall]		fraction of dataset to be used for training in the overall estimation module.
// [-c  | --n_boots_coarse]		number of bootstraps to use in the coarse lasso sweep.
// [-b	| --n_boots_sel]		number of bootstraps to use in the selection module (dense lasso sweep).
// [-e	| --n_boots_est]		number of bootstraps to use in the estimation module.
// [-g	| --bagging_options]		equal to 1: for each bootstrap sample, find the regularization
// 			                       parameter that gave the best results
//				        equal to 2: average estimates across bootstraps, and then find the
//			                       regularization parameter that gives the best results
// [-a	| --use_admm ]			flag indicating whether to use the ADMM algorithm.
// [-f  | --file ] ARG (std::string)   	Input File containing matrix and response vector
// [-v	| --verbose ]			verbose option
// [-d 	| --debug ]  			Debug option  boolean
// [-n  | --n_groups ]			Number of groups the or parallel bootstraps.
// [--n_minigroups ]			Number of parallel lambda executions.
// [--n_coarse ]			Number of parallel coarse bootstraps.
// [--n_minicoarse ]			Number of parallel coarse lambda.
// [--max_iter ] 			Maximum number of iterations for Lasso ADMM
// [--reltol ]				RELTOL hyperparameter variable for Lasso ADMM
// [--abstol ]				ABSTOL hyperparameter variable for Lasso ADMM
// [--rho ]				rho variable for Lasso ADMM			
// TODO: add normalize function equivalent to sklearn _preprocess_data.
//---------------------------------------------------------

class CommandLineOptions
{
public:
	enum statusReturn_e
	{
		OPTS_SUCCESS,
		OPTS_VERSION,
		OPTS_HELP,
		OPTS_FAILURE
	};

	CommandLineOptions();
	~CommandLineOptions();
	statusReturn_e parse( int argc, char* argv[] );
	
	inline const std::string & getInputFile() const;
	inline const std::string & getOutputFile1() const;
	inline const std::string & getOutputFile2() const;
	inline const std::string & getDatasetMatrix() const;
	inline const std::string & getDatasetVector() const;
	inline int getLambdas() {return myn_lambdas; };
	inline float getSelectionThreshold() {return myselection_thres_frac; }; 
	inline float getTrainSelection() {return mytrain_frac_sel; };
	inline float getTrainEstimation() {return mytrain_frac_est; };
	inline float getTrainOverall() {return mytrain_frac_overall; };
	inline int getBootsCoarse() {return myn_boots_coarse; };
	inline int getBootsSel() {return myn_boots_sel; };
	inline int getBootsEst() {return myn_boots_est; };
	inline int getBaggingOption() {return mybagging_options; };
	inline bool getUseAdmm() {return myuse_admm; };
	inline bool getVerbose() {return myverbose; };
	inline bool getDebug() {return mydebug; };
	inline int getnGroups() {return myn_groups; }
	inline int getnCoarse() {return myn_coarse; }
	inline int getnMiniCoarse() {return myn_minicoarse; }
	inline int getMiniGroups() {return myn_minigroups;}
	inline int getMAXITER() {return mymax_iter;}
	inline float getRELTOL() {return myreltol;}	
	inline float getABSTOL() {return myabstol;}
	inline float getRho()	{return myrho; }

protected:
	void setup();
	bool validateFiles();

private:
	//Not implemented and not for usage:
	CommandLineOptions( const CommandLineOptions &rhs ); 

	po::options_description myOptions;
	std::string myInputFile;
	std::string myOutputFile1;
	std::string myOutputFile2;
	std::string mydata_matrix;
	std::string mydata_vector;
	int myn_lambdas;
	float myselection_thres_frac;
	float mytrain_frac_sel;
	float mytrain_frac_est;
	float mytrain_frac_overall;
	int myn_boots_coarse;
	int myn_boots_sel;
	int myn_boots_est;
	int mybagging_options;
	bool myuse_admm;
	bool myverbose;
	bool mydebug;
	int myn_groups;
	int myn_coarse;
	int myn_minicoarse;
	int myn_minigroups;
	int mymax_iter;
	float myreltol;
	float myabstol;
	float myrho;
	
};

inline
const std::string & CommandLineOptions::getInputFile() const
{
	static const std::string emptyString;
	return ( 0 < myInputFile.size() ) ? myInputFile : emptyString;
}


inline
const std::string & CommandLineOptions::getOutputFile1() const
{
        static const std::string emptyString;
        return ( 0 < myOutputFile1.size() ) ? myOutputFile1 : emptyString;
}

inline
const std::string & CommandLineOptions::getOutputFile2() const
{
        static const std::string emptyString;
        return ( 0 < myOutputFile2.size() ) ? myOutputFile2 : emptyString;
}

inline
const std::string & CommandLineOptions::getDatasetMatrix() const
{
        static const std::string emptyString;
        return ( 0 < mydata_matrix.size() ) ? mydata_matrix : emptyString;
}

inline
const std::string & CommandLineOptions::getDatasetVector() const
{
        static const std::string emptyString;
        return ( 0 < mydata_vector.size() ) ? mydata_vector : emptyString;
}


#endif // _COMMAND_LINE_OPTONS_H_
