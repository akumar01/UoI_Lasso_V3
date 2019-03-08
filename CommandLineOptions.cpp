#include <boost/algorithm/string/trim.hpp>
#include <iostream>
#include <string>
#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
namespace fs = boost::filesystem;

#include "CommandLineOptions.h"

CommandLineOptions::CommandLineOptions():
	myOptions(""),
	myInputFile(""),
	myOutputFile1(""),
	myOutputFile2(""),
	mydata_matrix(""),
	mydata_vector(""),
	myn_lambdas( 0xffffffff ),
	myselection_thres_frac( -9999999.9999999 ),
	mytrain_frac_sel( -9999999.9999999 ),
	mytrain_frac_est( -9999999.9999999 ),
	mytrain_frac_overall( -9999999.9999999 ),
	myn_boots_coarse( 0xffffffff ),
	myn_boots_sel( 0xffffffff ),
	myn_boots_est( 0xffffffff ),
	mybagging_options( 0xffffffff ),
	myuse_admm( false ),
	myverbose( false ),
	mydebug( false ),
	myn_groups( 0xffffffff ),
	myn_coarse( 0xffffffff ),
	myn_minicoarse( 0xffffffff ),
	myn_minigroups( 0xffffffff  ),
	mymax_iter( 0xffffffff ),
	myreltol( -9999999.9999999 ),
	myabstol( -9999999.9999999 ),
	myrho( -9999999.9999999 )
{
	setup();

}


CommandLineOptions::~CommandLineOptions()
{

}


void CommandLineOptions::setup() 
{

	po::options_description options("Program Options");
	options.add_options()
		( "help,h", "Display help menu.")
		( "version,V", "Display Program version number")
		( "file,f",po::value<std::string>(&myInputFile), "Input File containing matrix and response vector")
		( "output_coef",po::value<std::string>(&myOutputFile1)->default_value( "coef.h5" ), "Output File containing final coefficients (coef_)" )
		( "output_scores",po::value<std::string>(&myOutputFile2)->default_value( "scores.h5" ), "Output File containing final scores (scores_)" )
		( "dataset_matrix",po::value<std::string>(&mydata_matrix)->default_value( "/X/data" ), "String of data matrix name and structure in the hdf5 file" )
		( "dataset_vector",po::value<std::string>(&mydata_vector)->default_value( "/y/data" ), "String of data vector name and structure in the hdf5 file" )
		( "n_lambdas,l",po::value<int>(&myn_lambdas)->default_value( 48 ), "number of L1 penalty values to compare across (effectively sets the hyperparameter sweep)"  )
		( "selection_thres_frac,s", po::value<float>(&myselection_thres_frac)->default_value( 1.0 ), "used for soft thresholding in the selection step. normally, UoI-Lasso requires regressors to be selected in _all_ bootstraps to be selected for use in the estimation module. this requirement can be softened with this variable, by requiring that a regressor appear in selection_thres_frac of the bootstraps." )
		("train_frac_sel,t", po::value<float>(&mytrain_frac_sel)->default_value( 0.8 ), "fraction of dataset to be used for training in the selection module.")
		( "train_frac_est,T", po::value<float>(&mytrain_frac_est)->default_value(0.8), "fraction of dataset to be used for training in each bootstrap in the estimation module." )
		( "train_frac_overall,O", po::value<float>(&mytrain_frac_overall)->default_value(0.9), "fraction of dataset to be used for training in the overall estimation module." )
		( "n_boots_coarse,c", po::value<int>(&myn_boots_coarse)->default_value(10), "number of bootstraps to use in the coarse lasso sweep." )
		( "n_boots_sel,b", po::value<int>(&myn_boots_sel)->default_value(48), "number of bootstraps to use in the selection module (dense lasso sweep)." )	
		( "n_boots_est,e", po::value<int>(&myn_boots_est)->default_value(48), "number of bootstraps to use in the estimation module." )
		( "bagging_options,g", po::value<int>(&mybagging_options)->default_value(1), "equal to 1: for each bootstrap sample, find the regularization parameter that gave the best results equal to 2: average estimates across bootstraps, and then find the regularization parameter that gives the best results" )
		( "use_admm,a", po::bool_switch(&myuse_admm)->default_value(true), "Toggle. Flag indicating whether to use the ADMM algorithm.")
		( "verbose,v", po::bool_switch(&myverbose)->default_value(false), "Verbose option")
		( "debug,d", po::bool_switch(&mydebug)->default_value( false ), "Debug option boolean")
		( "n_groups,n", po::value<int>(&myn_groups)->default_value( 1 ), "Number of groups for level 1 parallelism.")
		("n_minigroups", po::value<int>(&myn_minigroups)->default_value( 1 ), "Number of groups for level 2 parallelism." )
		("n_coarse", po::value<int>(&myn_coarse)->default_value( 1 ), "Number of groups for level 1 parallelism for coarse sweep." )
		("n_minicoarse", po::value<int>(&myn_minicoarse)->default_value( 1 ), "Number of groups for level 2 parallelism for coarse sweep." )
		( "max_iter", po::value<int>(&mymax_iter)->default_value( 50 ), "Maximum number of iterations for Lasso ADMM")
		( "reltol", po::value<float>(&myreltol)->default_value( 1e-2 ), "RELTOL hyperparameter variable for Lasso ADMM")
		( "abstol", po::value<float>(&myabstol)->default_value( 1e-4 ), "ABSTOL hyperparameter variable for Lasso ADMM")
		( "rho", po::value<float>(&myrho)->default_value( 1.0 ), "rho variable for Lasso ADMM "); 

	myOptions.add( options );

}

CommandLineOptions::statusReturn_e CommandLineOptions::parse( int argc, char* argv[] )
{
	statusReturn_e ret = OPTS_SUCCESS;
	
	po::variables_map varMap;
	char filename[2000];
	
	try
	{
		po::store( po::parse_command_line( argc, argv, myOptions ), varMap);
		po::notify( varMap );
		
		//Help option
		if( varMap.count( "help" ) )
		{
			std::cout << myOptions << std::endl;
			return OPTS_HELP;
		}

		if( varMap.count( "version" ))
		{
			return OPTS_VERSION;	
		}

		//Input file error check
		if( !(0 < varMap.count( "file" ) ) )
		{
			std::cout << "ERROR -- Input File must be specified!!!" << std::endl;
			std::cout << myOptions << std::endl;
			return OPTS_FAILURE;
		}
	
		else
		{
			boost::algorithm::trim( myInputFile );

			realpath( myInputFile.c_str(), filename );
			myInputFile = filename;

			ret = validateFiles() ? OPTS_SUCCESS : OPTS_FAILURE;
		}
	}
	
	catch( std::exception &e )
	{
		std::cout << "ERROR -- parsing error: " << e.what() << std::endl;
		ret = OPTS_FAILURE;
	}
	catch( ... )
	{
		std::cout << "ERROR -- parsing error (unknown type) " << std::endl;	
		ret = OPTS_FAILURE;
	}
	
	return ret;

}

bool CommandLineOptions::validateFiles()
{
	if( !boost::filesystem::is_regular_file(myInputFile))
	{
		std::cout << "ERROR -- Input File provided does not exist! [" << myInputFile << "]" << std::endl;
		return false;
	}
	
	return true;
}
	
