#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "hdf5.h"
#include "manage-data.h"

int get_rows (char *infile, char *dataset) {
   
    int rows;

    hid_t file_id, dataset_id;
    file_id = H5Fopen(infile, H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset_id = H5Dopen(file_id, dataset, H5P_DEFAULT);
    hid_t dspace = H5Dget_space(dataset_id);
    const int ndims = H5Sget_simple_extent_ndims(dspace);
    hsize_t dims[ndims];
    H5Sget_simple_extent_dims(dspace, dims, NULL);
    rows = dims[0];

    H5Dclose(dataset_id);
    H5Fclose(file_id);

    return rows;
}

int get_cols (char *infile, char *dataset) { 

    int cols;

    hid_t file_id, dataset_id;
    file_id = H5Fopen(infile, H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset_id = H5Dopen(file_id, dataset, H5P_DEFAULT);
    hid_t dspace = H5Dget_space(dataset_id);
    const int ndims = H5Sget_simple_extent_ndims(dspace);
    hsize_t dims[ndims];
    H5Sget_simple_extent_dims(dspace, dims, NULL);
    cols = dims[1];

    H5Dclose(dataset_id);
    H5Fclose(file_id);

    return cols;
}

/*void get_matrix (int Matrows, int Matcols, int Totalrows, MPI_Comm comm, int mpi_rank, float * Mat, char dataset[], char infile[]) { 
   hid_t       file_id, dset_id;    
   hid_t       filespace, memspace;   
   hsize_t     count[2];            
   hsize_t     offset[2];
   hsize_t     dimsf[2]; 
   hid_t       plist_id;            
   herr_t      status;

   plist_id = H5Pcreate(H5P_FILE_ACCESS);
   H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);
   file_id = H5Fopen(infile, H5F_ACC_RDONLY, plist_id);

   dimsf[0] = Totalrows;
   dimsf[1] = Matcols;
   filespace = H5Screate_simple(2, dimsf, NULL); 
 
   plist_id = H5Pcreate(H5P_DATASET_XFER);
   status = H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

   dset_id = H5Dopen2(file_id, dataset, plist_id);
   //dset_id = H5Dcreate(file_id, dataset, H5T_NATIVE_FLOAT, filespace,
   //		H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   H5Sclose(filespace);

   count[0] = Matrows;
   count[1] = Matcols;
   offset[0] = mpi_rank * count[0];
   offset[1] = 0;
   memspace = H5Screate_simple(2, count, NULL);

   filespace = H5Dget_space(dset_id);
   H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);

   //plist_id = H5Pcreate(H5P_DATASET_XFER);
   //status = H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

   status = H5Dread(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, Mat); 
}*/ 

float* get_matrix (int Matrows, int Matcols, int Totalrows, MPI_Comm comm, int mpi_rank, char *dataset, char *infile) {

   hid_t       file_id, dset_id;    
   hid_t       dataspace, memspace;   
   hsize_t     count[2], count_out[2];            
   hsize_t     offset[2], offset_out[2];
   hsize_t     dimsf[2]; 
   hid_t       plist_id;            
   herr_t      status;
  
   float *Mat;
   Mat = (float*) malloc (Matrows * Matcols * sizeof(float));

   plist_id = H5Pcreate(H5P_FILE_ACCESS);
   H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);
   file_id = H5Fopen(infile, H5F_ACC_RDONLY, plist_id);
   dset_id = H5Dopen(file_id, dataset, H5P_DEFAULT);
   dataspace = H5Dget_space (dset_id); 

   /*Define hyperslab in dataset*/

   count[0] = Matrows;
   count[1] = Matcols;
   offset[0] = mpi_rank * count[0];
   offset[1] = 0;
   status = H5Sselect_hyperslab (dataspace, H5S_SELECT_SET, offset, NULL, count, NULL);


   /*Define memory space*/ 

   dimsf[0] = Matrows;
   dimsf[1] = Matcols; 
   memspace =  H5Screate_simple (2, dimsf, NULL);  


   /* Memory hyperslab*/ 
   offset_out[0] = 0; 
   offset_out[1] = 0;
   count_out[0] = Matrows;
   count_out[1] = Matcols;
   status = H5Sselect_hyperslab (memspace, H5S_SELECT_SET, offset_out, NULL, count_out, NULL);

   plist_id = H5Pcreate(H5P_DATASET_XFER);
   H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

   status = H5Dread (dset_id, H5T_NATIVE_FLOAT, memspace, dataspace, plist_id, Mat);

   return Mat; 

}


float* get_array (int Arrlen, int Totalrows, MPI_Comm comm, int mpi_rank, char *dataset, char *infile) {

   hid_t       file_id, dset_id;
   hid_t       dataspace, memspace;
   hsize_t     count[2], count_out[2];
   hsize_t     offset[2], offset_out[2];
   hsize_t     dimsf[2];
   hid_t       plist_id;
   herr_t      status;


   plist_id = H5Pcreate(H5P_FILE_ACCESS);
   H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);
   file_id = H5Fopen(infile, H5F_ACC_RDONLY, plist_id);
   dset_id = H5Dopen(file_id, dataset, H5P_DEFAULT);
   dataspace = H5Dget_space (dset_id);

   float* Arr;
   Arr = (float*) malloc (Arrlen * sizeof(float));

   /*Define hyperslab in dataset*/

   count[0] = Arrlen;
   count[1] = 1;
   offset[0] = mpi_rank * count[0];
   offset[1] = 0; 
   H5Sselect_hyperslab (dataspace, H5S_SELECT_SET, offset, NULL, count, NULL);


   /*Define memory space*/

   dimsf[0] = Arrlen;
   dimsf[1] = 1; 
   memspace =  H5Screate_simple (2, dimsf, NULL);


   /* Memory hyperslab*/
   offset_out[0] = 0;
   offset_out[1] = 0;
   count_out[0] = Arrlen;
   count_out[1] = 1;
   H5Sselect_hyperslab (memspace, H5S_SELECT_SET, offset_out, NULL, count_out, NULL);

   plist_id = H5Pcreate(H5P_DATASET_XFER);
   H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

   status = H5Dread (dset_id, H5T_NATIVE_FLOAT, memspace, dataspace, plist_id, Arr);
   return Arr;

}

float*  get_array1 (int Arrlen, int Totalrows, MPI_Comm comm, int mpi_rank, char dataset[], char infile[]) {

   hid_t       file_id, dset_id;
   hid_t       dataspace, memspace;
   hsize_t     count[1], count_out[1];
   hsize_t     offset[1], offset_out[1];
   hsize_t     dimsf[1];
   hid_t       plist_id;
   herr_t      status;

   float *Arr;
   Arr = (float *) malloc (Arrlen * sizeof(float));

   plist_id = H5Pcreate(H5P_FILE_ACCESS);
   H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);
   file_id = H5Fopen(infile, H5F_ACC_RDONLY, plist_id);
   dset_id = H5Dopen(file_id, dataset, H5P_DEFAULT);
   dataspace = H5Dget_space (dset_id);

   /*Define hyperslab in dataset*/

   count[0] = Arrlen;
   offset[0] = mpi_rank * count[0];
   H5Sselect_hyperslab (dataspace, H5S_SELECT_SET, offset, NULL, count, NULL);


   /*Define memory space*/

   dimsf[0] = Arrlen;
   memspace =  H5Screate_simple (1, dimsf, NULL);


   /* Memory hyperslab*/
   offset_out[0] = 0;
   count_out[0] = Arrlen;
   H5Sselect_hyperslab (memspace, H5S_SELECT_SET, offset_out, NULL, count_out, NULL);

   plist_id = H5Pcreate(H5P_DATASET_XFER);
   H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

   status = H5Dread (dset_id, H5T_NATIVE_FLOAT, memspace, dataspace, plist_id, Arr);

   return Arr;

}

/*void get_array (int array_len, int Totalrows, MPI_Comm comm, int mpi_rank, float * Arr, char dataset[], char infile[]) {

   hid_t       file_id, dset_id;
   hid_t       filespace, memspace;
   hsize_t     count[1];
   hsize_t     offset[1];
   hsize_t     dimsf[1];
   hid_t       plist_id;
   herr_t      status;

   plist_id = H5Pcreate(H5P_FILE_ACCESS);
   H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);
   file_id = H5Fopen(infile, H5F_ACC_RDONLY, plist_id);
   
   dimsf[0] = Totalrows;
   filespace = H5Screate_simple(1, dimsf, NULL);

   plist_id = H5Pcreate(H5P_DATASET_XFER);
   status = H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    dset_id = H5Dopen2(file_id, dataset, plist_id);

   //dset_id = H5Dopen2(file_id, dataset, H5P_DEFAULT);
   //dset_id = H5Dcreate(file_id, dataset, H5T_NATIVE_FLOAT, filespace,
   //                     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

   count[0] = array_len;
   offset[0] = mpi_rank * count[0];
   memspace = H5Screate_simple(1, count, NULL);

   filespace = H5Dget_space(dset_id);
   H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);

   //plist_id = H5Pcreate(H5P_DATASET_XFER);
   //status = H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

   status = H5Dread(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, Arr);
}*/


/*void combine_matrix (float * Mat, float *Arr, float * Out, int rows, int cols) {

   int i, j;
   //copy Mat (m X n) into Out
   for (i=0;i<rows;i++)
	for (j=0;j<cols+1;j++) {

	   if (j==cols)
		 *(Out + i*(cols+1) + j) = Arr[i];
	   else
	   	*(Out + i*(cols+1) + j) = *(Mat + i*cols + j);

	}
}*/


/*float* split_matrix (float *Mat, float *Out, int rows, int cols, int this_rank) {

   int i, j;
   float *Arr;
   Arr = (float *)malloc (rows * sizeof (float)); 

  int lenj;
 FILE *fp1;
 fp1 = fopen("y_in.txt", "w");

 for (leni =0; lenj < rows; leni++) {
      fprintf(fp, "%lf\n", Arr[i]);
   }

fclose (fp);
  
   for (i=0;i<rows;i++)
        for (j=0;j<cols+1;j++) {

           if (j==cols)
                 Arr[i] = *(Mat + i*(cols+1) + j); 
           else
                *(Out + i*(cols) + j) = *(Mat + i*cols + j);

        }

if (this_rank == 0) {
	int leni;
 FILE *fp;
 fp = fopen("y_inside.txt", "w");

 for (leni =0; leni < rows; leni++) {
      fprintf(fp, "%lf\n", Arr[leni]);
   }

fclose (fp);
   
}

return Arr; 

}*/

/*void get_train (float * Mat, float * Vec, float * Mat_train, float * Vec_train, int train,  int cols) {

   int i, j;
   float sum=0; 
 
   for (i=0;i<train;i++)
	for (j=0;j<cols;j++)
	 	*(Mat_train + i*cols + j) = *(Mat + i*cols + j); 


  for (i=0;i<train;i++) 
	sum += (float) *(Vec + i);
   
  //printf("sum : %f\n", sum);  
  float avg = (sum/train);
  //printf("avg : %f\n", avg); 

  for (i=0;i<train;i++)
	Vec_train[i] = Vec[i] - avg;   	
}*/ 


void write_1D(hid_t id, int DIMS0, float *data,hid_t plist,  char name[]){

  hid_t dataspace_id, dataset_id;
  herr_t status3;
  hsize_t dims[1];
  hid_t       filespace, memspace;

  dims[0] = DIMS0;
  dataspace_id = H5Screate_simple(1, dims, NULL);

   /* Create a dataset in group "MyGroup". */
  dataset_id = H5Dcreate(id, name,  H5T_IEEE_F64LE, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  filespace = H5Screate_simple(1, dims, NULL); 
  memspace  = H5Screate_simple(1, dims, NULL);
  status3 = H5Dwrite(dataset_id,  H5T_NATIVE_FLOAT, memspace, filespace, plist,
                     &data);
  status3 = H5Dclose(dataset_id);
  status3 = H5Sclose(dataspace_id);
}


void write_2D(hid_t id, int DIMS0, int DIMS1, float *data, hid_t plist, char name[]){
   
  hid_t dataspace_id, dataset_id;
  herr_t status2;
  hsize_t dims[2]; 
  hid_t       filespace, memspace;

  dims[0] = DIMS0;
  dims[1] = DIMS1;
  dataspace_id = H5Screate_simple(2, dims, NULL);

   /* Create a dataset in group "MyGroup". */
  dataset_id = H5Dcreate(id, name,  H5T_IEEE_F64LE, dataspace_id,
                          H5P_DEFAULT,H5P_DEFAULT, H5P_DEFAULT);
  filespace = H5Screate_simple(2, dims, NULL);    
  memspace  = H5Screate_simple(2, dims, NULL);
  status2 = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT,
                     &data);
  status2 = H5Dclose(dataset_id);
  status2 = H5Sclose(dataspace_id);
}


/*void write_attr(hid_t file_id, int attr_data, char attr[]){

  hid_t       dataset_id, attribute_id, dataspace_id;
  hsize_t     dims;
  herr_t      status1;
 // hid_t       filespace, memspace;

  dims = 1;
  dataspace_id = H5Screate_simple(1, &dims, NULL);
  dataset_id = H5Dcreate(file_id, "Attr", H5T_STD_I32BE, dataspace_id,H5P_DEFAULT,H5P_DEFAULT, H5P_DEFAULT);
  attribute_id = H5Acreate2 (dataset_id, attr, H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status1 = H5Awrite(attribute_id, H5T_NATIVE_INT, &attr_data);
  status1 = H5Aclose(attribute_id);
  status1 = H5Dclose(dataset_id); 
  status1 = H5Sclose(dataspace_id);

}*/


void write_D(hid_t id, float data, hid_t plist, char name[]){

  hid_t dataspace_id, dataset_id;
  herr_t status;
  hsize_t dims[1];
  hid_t       filespace, memspace;

  dims[0] = 1;
  dataspace_id = H5Screate_simple(1, dims, NULL);

   /* Create a dataset in group "MyGroup". */
  dataset_id = H5Dcreate(id, name,  H5T_IEEE_F64LE, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  filespace = H5Screate_simple(1, dims, NULL);    
  memspace  = H5Screate_simple(1, dims, NULL);
  status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, memspace, filespace, 
                    plist, &data);
  status = H5Dclose(dataset_id);
  status = H5Sclose(dataspace_id);
}
/*void write_attr(hid_t file_id, float attr_data, char attr[]){

  hid_t       dataset_id, attribute_id, dataspace_id; 
  hsize_t     dims;
  herr_t      status;

  dims = 1;
  dataspace_id = H5Screate_simple(1, &dims, NULL);
  dataset_id = H5Dcreate(file_id, "Attr", H5T_STD_I32BE, dataspace_id,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  attribute_id = H5Acreate2 (dataset_id, attr, H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id, H5T_NATIVE_FLOAT, attr_data);
  status = H5Aclose(attribute_id);
  status = H5Dclose(dataset_id);
  status = H5Sclose(dataspace_id);

}*/

/*void write_data (char OutFile[], int maxBoot, int bgdOpt, int nrnd, float cvlfrct, float rndfrct, float rndfrctL, int nbootE, int nbootS, int nMP, int seed, int m, int n, 
                float end_loadTime, float end_distTime, float end_commTime, float end_las1Time, float end_las2Time, float end_olsTime, 
                 float *B0, float *R2m0, float *lamb0, float *B, float *R2m, float *lambL, float *sprt, float *Bgd_m, float *R2_m, int rsd_size_, float *rsd, float *bic, MPI_Comm comm ) {

   hid_t       file_id, group_id, plist_id;
   herr_t 	status; 
   int rank_w, size_w; 

   MPI_Comm_rank (comm, &rank_w);
   MPI_Comm_size(comm, &size_w);

   plist_id = H5Pcreate(H5P_FILE_ACCESS);
   H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);

   file_id = H5Fcreate(OutFile, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
   H5PcGlose(plist_id);

  
    plist_id = H5Pcreate(H5P_DATASET_XFER);
   H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT); 

   if (rank_w==0) {
 
   group_id = H5Gcreate(file_id, "Attributes", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

   write_D(file_id, bgdOpt, plist_id, "bgdOpt"); 
   write_D(file_id, nrnd, plist_id, "nrnd");
   write_D(file_id, cvlfrct,plist_id,  "cvlfrct");
   write_D(file_id, rndfrct, plist_id, "rndfrct");
   write_D(file_id, rndfrctL, plist_id, "rndfrctL");
   write_D(file_id, nbootE, plist_id, "nbootE");
   write_D(file_id, nbootS, plist_id, "nbootS");
   write_D(file_id, nMP, plist_id, "nMP");
   write_D(file_id, seed, plist_id, "seed");
   write_D(file_id, m, plist_id, "m");
   write_D(file_id, n, plist_id, "n");

   printf("completed writing attr\n"); 

   group_id = H5Gcreate(file_id, "Time", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

   write_D(group_id, end_loadTime, plist_id, "loadTime");
   write_D(group_id, end_distTime, plist_id, "distTime");
   write_D(group_id, end_commTime, plist_id, "commTime");
   write_D(group_id, end_las1Time, plist_id, "las1time");
   write_D(group_id, end_las2Time, plist_id, "las2Time");
   write_D(group_id, end_olsTime, plist_id, "EstimationTime"); 
   status = H5Gclose(group_id);

    printf("completed writing time\n");
 
   group_id = H5Gcreate(file_id, "lasso", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

   write_2D(group_id, nbootS, n, B0, plist_id, "B0");
   write_1D(group_id, nbootS, R2m0, plist_id, "R2m0"); 
   write_1D(group_id, nbootS,lamb0, plist_id, "lamb0");
   write_2D(group_id, nbootS, n, B, plist_id, "B"); 
   write_1D(group_id, nbootS, R2m, plist_id ,"R2m");
   write_1D(group_id, nbootS,lambL, plist_id, "lambL");
   status = H5Gclose(group_id);


    printf("completed writing lasso\n");

   group_id = H5Gcreate(file_id, "uoi", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); 

   write_2D(group_id, nbootS, n, sprt, plist_id, "sprt");  
   write_2D(group_id, nrnd, n, Bgd_m, plist_id, "Bgd");
   write_1D(group_id, nrnd, R2_m, plist_id, "R2");
   write_2D(group_id, nrnd, rsd_size_, rsd, plist_id, "rsd");
   write_1D(group_id, nrnd, bic, plist_id, "bic");

    printf("completed writing uoi\n");
    status = H5Gclose(group_id);
 }
   status = H5Pclose(plist_id);
   status = H5Fclose(file_id);  
 
   
}*/  



void write_output(char OutFile[], int nrnd, int n, float *Bgd_m, float *R2_m, int rsd_size_, float *rsd, float *bic, MPI_Comm comm ) {

     /*
     * HDF5 APIs definitions
     */ 	
    hid_t       file_id, dset_id, group_id, attribute_id;         /* file and dataset identifiers */
    hid_t       filespace, memspace;      /* file and memory dataspace identifiers */
    hsize_t     dimsf[2], dims[1];                 /* dataset dimensions */
    int         data[1];                    /* pointer to data buffer to write */
    hsize_t	count[2], counts[1];	          /* hyperslab selection parameters */
    hsize_t	offset[2], offsets[1];
    hid_t	plist_id;                 /* property list identifier */
    int         i;
    herr_t	status;


    int mpi_size, mpi_rank;
    MPI_Info info  = MPI_INFO_NULL;
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &mpi_rank);   
 
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, comm, info);
    file_id = H5Fcreate(OutFile, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Pclose(plist_id);

    /* Write Bgd */ 

    dimsf[0] = nrnd*mpi_size;
    dimsf[1] = n;
    filespace = H5Screate_simple(2, dimsf, NULL); 

    group_id = H5Gcreate(file_id, "UOI", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); 
    dset_id = H5Dcreate(group_id, "Bgd", H5T_NATIVE_FLOAT, filespace,
			H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    count[0] = dimsf[0]/mpi_size;
    count[1] = dimsf[1];
    offset[0] = mpi_rank * count[0];
    offset[1] = 0;
    memspace = H5Screate_simple(2, count, NULL); 


    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
	
    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
    
    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
		      plist_id, Bgd_m); 

   /* Write RSD */ 

    dimsf[0] = nrnd*mpi_size;
    dimsf[1] = rsd_size_;
    filespace = H5Screate_simple(2, dimsf, NULL);

    dset_id = H5Dcreate(group_id, "rsd", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

     count[0] = dimsf[0]/mpi_size;
    count[1] = dimsf[1];
    offset[0] = mpi_rank * count[0];
    offset[1] = 0;
    memspace = H5Screate_simple(2, count, NULL);
 
    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, rsd);

   /* Write R2 */ 

    dims[0] = nrnd*mpi_size; 

    filespace = H5Screate_simple(1, dims, NULL);
    dset_id = H5Dcreate(group_id, "R2", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace); 

    counts[0] = dims[0]/mpi_size;
    offsets[0] = mpi_rank * counts[0]; 
     memspace = H5Screate_simple(1, counts, NULL);

    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, count, NULL);

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, R2_m);
   
    /*Write bic*/ 

    filespace = H5Screate_simple(1, dims, NULL);
    dset_id = H5Dcreate(group_id, "bic", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace); 

     memspace = H5Screate_simple(1, counts, NULL);

    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, count, NULL);

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, bic);


    H5Pclose(plist_id);
    H5Sclose(filespace);
    H5Sclose(memspace);
    H5Dclose(dset_id);
    H5Gclose(group_id);
   // H5Fflush(file_id, H5F_SCOPE_GLOBAL); 
   // H5Fclose(file_id);


}

void write_inter(char OutFile[], int nboot, int n, float *B0, float R20, float lambC, float *B, float R2m, float lambD, float *sprt_h, int m, int nbootE, int nrnd, float cvlfrct, float rndfrct, float rndfrctL, float end_read, float tmax, float dist_end, float redis_end, float lasso1_end, float lasso2_end, float est_end, float end_ols, MPI_Comm comm ) {

     /*
     * HDF5 APIs definitions
     */
    hid_t       file_id, dset_id, group_id, attribute_id;         /* file and dataset identifiers */
    hid_t       filespace, memspace;      /* file and memory dataspace identifiers */
    hsize_t     dimsf[2], dims[1];                 /* dataset dimensions */
    int         data[1];                    /* pointer to data buffer to write */
    hsize_t     count[2], counts[1];              /* hyperslab selection parameters */
    hsize_t     offset[2], offsets[1];
    hid_t       plist_id;                 /* property list identifier */
    int         i;
    herr_t      status;


    int mpi_size, mpi_rank;
    MPI_Info info  = MPI_INFO_NULL;
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &mpi_rank);

    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, comm, info);
    //file_id = H5Fopen(OutFile, H5F_ACC_RDWR, H5P_DEFAULT);
    file_id = H5Fcreate(OutFile, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Pclose(plist_id);


    /* Write B0 */

    dimsf[0] = nboot;
    dimsf[1] = n;
    filespace = H5Screate_simple(2, dimsf, NULL);

    group_id = H5Gcreate(file_id, "lasso", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_id = H5Dcreate(group_id, "B0", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    count[0] = dimsf[0]/mpi_size;
    count[1] = dimsf[1];
    offset[0] = mpi_rank * count[0];
    offset[1] = 0;
    memspace = H5Screate_simple(2, count, NULL);


    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);

    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, B0);


   /* Write R2 */

    dims[0] = mpi_size;

    filespace = H5Screate_simple(1, dims, NULL);
    dset_id = H5Dcreate(group_id, "R2m0", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    counts[0] = dims[0]/mpi_size;
    offsets[0] = mpi_rank * counts[0];
     memspace = H5Screate_simple(1, counts, NULL);

    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, count, NULL);

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, &R20);


   /* Write lamb0 */

    dims[0] = mpi_size;

    filespace = H5Screate_simple(1, dims, NULL);
    dset_id = H5Dcreate(group_id, "lamb0", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    counts[0] = dims[0]/mpi_size;
    offsets[0] = mpi_rank * counts[0];
     memspace = H5Screate_simple(1, counts, NULL);

    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, count, NULL);

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, &lambC);


   /* Write B */

    dimsf[0] = nboot;
    dimsf[1] = n;
    filespace = H5Screate_simple(2, dimsf, NULL);
    dset_id = H5Dcreate(group_id, "B", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    count[0] = dimsf[0]/mpi_size;
    count[1] = dimsf[1];
    offset[0] = mpi_rank * count[0];
    offset[1] = 0;
    memspace = H5Screate_simple(2, count, NULL);


    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);

    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, B);


   /* Write R2m */

    dims[0] = mpi_size;

    filespace = H5Screate_simple(1, dims, NULL);
    dset_id = H5Dcreate(group_id, "R2m", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    counts[0] = dims[0]/mpi_size;
    offsets[0] = mpi_rank * counts[0];
     memspace = H5Screate_simple(1, counts, NULL);

    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, count, NULL);

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, &R2m);


   /* Write lambL */

    dims[0] = mpi_size;

    filespace = H5Screate_simple(1, dims, NULL);
    dset_id = H5Dcreate(group_id, "lambL", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    counts[0] = dims[0]/mpi_size;
    offsets[0] = mpi_rank * counts[0];
     memspace = H5Screate_simple(1, counts, NULL);

    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, count, NULL);

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, &lambD);


   /* Write sprt */

    dimsf[0] = nboot*mpi_size;
    dimsf[1] = n;
    filespace = H5Screate_simple(2, dimsf, NULL);
    dset_id = H5Dcreate(group_id, "sprt", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    count[0] = dimsf[0]/mpi_size;
    count[1] = dimsf[1];
    offset[0] = mpi_rank * count[0];
    offset[1] = 0;
    memspace = H5Screate_simple(2, count, NULL);


    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);

    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, sprt_h);
    
    H5Gclose(group_id);

/*

   // write attributes
   // nbootS
    group_id = H5Gcreate(file_id, "Attributes", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); 
    dims[0] = mpi_size;

    filespace = H5Screate_simple(1, dims, NULL);
    dset_id = H5Dcreate(group_id, "nbootS", H5T_NATIVE_INT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    counts[0] = dims[0]/mpi_size;
    offsets[0] = mpi_rank * counts[0];
     memspace = H5Screate_simple(1, counts, NULL);

    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, count, NULL);

    status = H5Dwrite(dset_id, H5T_NATIVE_INT, memspace, filespace,
                      plist_id, &nboot);

 
  //nbootE
   dims[0] = 1;

    filespace = H5Screate_simple(1, dims, NULL);
    dset_id = H5Dcreate(group_id, "nbootE", H5T_NATIVE_INT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    counts[0] = dims[0]/mpi_size;
    offsets[0] =  mpi_rank * counts[0];
     memspace = H5Screate_simple(1, counts, NULL);

    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, count, NULL);

    status = H5Dwrite(dset_id, H5T_NATIVE_INT, memspace, filespace,
                      plist_id, &nboot); 
  
    
   //m

   dims[0] = 1;

    filespace = H5Screate_simple(1, dims, NULL);
    dset_id = H5Dcreate(group_id, "m", H5T_NATIVE_INT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    counts[0] = dims[0]/mpi_size;
    offsets[0] = mpi_rank * counts[0];
     memspace = H5Screate_simple(1, counts, NULL);

    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, count, NULL);

    status = H5Dwrite(dset_id, H5T_NATIVE_INT, memspace, filespace,
                      plist_id, &m);



    //n

    filespace = H5Screate_simple(1, dims, NULL);
    dset_id = H5Dcreate(group_id, "n", H5T_NATIVE_INT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    counts[0] = dims[0]/mpi_size;
    offsets[0] =  mpi_rank * counts[0];
     memspace = H5Screate_simple(1, counts, NULL);

    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, count, NULL);

    status = H5Dwrite(dset_id, H5T_NATIVE_INT, memspace, filespace,
                      plist_id, &n);


    //nrnd 
    filespace = H5Screate_simple(1, dims, NULL);
    dset_id = H5Dcreate(group_id, "nrnd", H5T_NATIVE_INT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    counts[0] = dims[0]/mpi_size;
    offsets[0] =  mpi_rank * counts[0];
     memspace = H5Screate_simple(1, counts, NULL);

    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, count, NULL);

    status = H5Dwrite(dset_id, H5T_NATIVE_INT, memspace, filespace,
                      plist_id, &nrnd);

  
   // cvlfrct

    filespace = H5Screate_simple(1, dims, NULL);
    dset_id = H5Dcreate(group_id, "cvlfrct", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    counts[0] = dims[0]/mpi_size;
    offsets[0] =  mpi_rank * counts[0];
     memspace = H5Screate_simple(1, counts, NULL);

    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, count, NULL);

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, &cvlfrct);

    
   //rndfrct
    filespace = H5Screate_simple(1, dims, NULL);
    dset_id = H5Dcreate(group_id, "rndfrct", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    counts[0] = dims[0]/mpi_size;
    offsets[0] =  mpi_rank * counts[0];
     memspace = H5Screate_simple(1, counts, NULL);

    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, count, NULL);

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, &rndfrct);



    //rndfrctL
    filespace = H5Screate_simple(1, dims, NULL);
    dset_id = H5Dcreate(group_id, "rndfrctL", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    counts[0] = dims[0]/mpi_size;
    offsets[0] = mpi_rank * counts[0];
     memspace = H5Screate_simple(1, counts, NULL);

    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, count, NULL);

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, &rndfrctL);


   H5Gclose(group_id);

   //write time 
   group_id = H5Gcreate(file_id, "UoI_Times", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    filespace = H5Screate_simple(1, dims, NULL);
    dset_id = H5Dcreate(group_id, "load time", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    counts[0] = dims[0]/mpi_size;
    offsets[0] =  mpi_rank * counts[0];
     memspace = H5Screate_simple(1, counts, NULL);

    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, count, NULL);

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, &end_read); 


    filespace = H5Screate_simple(1, dims, NULL);
    dset_id = H5Dcreate(group_id, "comm time", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    counts[0] = dims[0]/mpi_size;
    offsets[0] =  mpi_rank * counts[0];
     memspace = H5Screate_simple(1, counts, NULL);

    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, count, NULL);

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, &tmax);


    filespace = H5Screate_simple(1, dims, NULL);
    dset_id = H5Dcreate(group_id, "dist time", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    counts[0] = dims[0]/mpi_size;
    offsets[0] = mpi_rank * counts[0];
     memspace = H5Screate_simple(1, counts, NULL);

    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, count, NULL);

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, &dist_end);


    filespace = H5Screate_simple(1, dims, NULL);
    dset_id = H5Dcreate(group_id, "redis time", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    counts[0] = dims[0]/mpi_size;
    offsets[0] = mpi_rank * counts[0];
     memspace = H5Screate_simple(1, counts, NULL);

    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, count, NULL);

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, &redis_end);


    float comptime=lasso1_end+lasso2_end+est_end; 

    filespace = H5Screate_simple(1, dims, NULL);
    dset_id = H5Dcreate(group_id, "computation time", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    counts[0] = dims[0]/mpi_size;
    offsets[0] = mpi_rank * counts[0];
     memspace = H5Screate_simple(1, counts, NULL);

    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, count, NULL);

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, &comptime);

  
    dset_id = H5Dcreate(group_id, "lasso1 time", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    counts[0] = dims[0]/mpi_size;
    offsets[0] = mpi_rank * counts[0];
     memspace = H5Screate_simple(1, counts, NULL);

    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, count, NULL);

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, &lasso1_end);



   dset_id = H5Dcreate(group_id, "lasso2 time", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    counts[0] = dims[0]/mpi_size;
    offsets[0] = mpi_rank * counts[0];
     memspace = H5Screate_simple(1, counts, NULL);

    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, count, NULL);

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, &lasso2_end);


   dset_id = H5Dcreate(group_id, "ols time", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    counts[0] = dims[0]/mpi_size;
    offsets[0] = mpi_rank * counts[0];
     memspace = H5Screate_simple(1, counts, NULL);

    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, count, NULL);

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, &end_ols);

   
   dset_id = H5Dcreate(group_id, "estimation time", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    counts[0] = dims[0]/mpi_size;
    offsets[0] = mpi_rank * counts[0];
     memspace = H5Screate_simple(1, counts, NULL);

    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, count, NULL);

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, &est_end); 
*/ 	

    H5Pclose(plist_id);
    H5Sclose(filespace);
    H5Sclose(memspace);
    H5Dclose(dset_id);
    H5Gclose(group_id);
   // H5Fflush(file_id, H5F_SCOPE_GLOBAL); 
    //H5Fclose(file_id);


}


void write_selections(char OutFile[], float *B, float *R2m,  float *lambda, float *sprt_in, int nboot, int nMP, int n, MPI_Comm comm) {

    /*
     * HDF5 APIs definitions
     */
    hid_t       file_id, dset_id, group_id, attribute_id;         /* file and dataset identifiers */
    hid_t       filespace, memspace;      /* file and memory dataspace identifiers */
    hsize_t     dimsf[2], dims[1];                 /* dataset dimensions */
    int         data[1];                    /* pointer to data buffer to write */
    hsize_t     count[2], counts[1];              /* hyperslab selection parameters */
    hsize_t     offset[2], offsets[1];
    hid_t       plist_id;                 /* property list identifier */
    int         i;
    herr_t      status;


    int mpi_size, mpi_rank;
    MPI_Info info  = MPI_INFO_NULL;
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &mpi_rank);

    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, comm, info);
    file_id = H5Fcreate(OutFile, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Pclose(plist_id);

    /* Write B0 */

    /*dimsf[0] = nboot*mpi_size;
    dimsf[1] = n;
    filespace = H5Screate_simple(2, dimsf, NULL);

    group_id = H5Gcreate(file_id, "lasso", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_id = H5Dcreate(group_id, "B0", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    count[0] = dimsf[0]/mpi_size;
    count[1] = dimsf[1];
    offset[0] = mpi_rank * count[0];
    offset[1] = 0;
    memspace = H5Screate_simple(2, count, NULL);


    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);

    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, B0);*/


    /* Write B */

    dimsf[0] = nboot*nMP;
    dimsf[1] = n;
    filespace = H5Screate_simple(2, dimsf, NULL);

    group_id = H5Gcreate(file_id, "lasso", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_id = H5Dcreate(group_id, "B", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    count[0] = dimsf[0]/mpi_size;
    count[1] = dimsf[1];
    offset[0] = mpi_rank * count[0];
    offset[1] = 0;
    memspace = H5Screate_simple(2, count, NULL);


    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);

    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, B);


    /* Write R20 */

    /*dimsf[0] = nboot*mpi_size;
    filespace = H5Screate_simple(1, dimsf, NULL);

    group_id = H5Gcreate(file_id, "lasso", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_id = H5Dcreate(group_id, "R2m0", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    count[0] = dimsf[0]/mpi_size;
    offset[0] = mpi_rank * count[0];
    memspace = H5Screate_simple(1, count, NULL);


    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);

    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                     plist_id, R20);*/


    /* Write R2m */

    dimsf[0] = nboot*nMP;
    filespace = H5Screate_simple(1, dimsf, NULL);

    dset_id = H5Dcreate(group_id, "R2m", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    count[0] = dimsf[0]/mpi_size;
    offset[0] = mpi_rank * count[0];
    memspace = H5Screate_simple(1, count, NULL);


    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);

    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, R2m);


     /* Write lamb0 */

    /*dimsf[0] = nboot*mpi_size;
    filespace = H5Screate_simple(1, dimsf, NULL);

    group_id = H5Gcreate(file_id, "lasso", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dset_id = H5Dcreate(group_id, "lamb0", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    count[0] = dimsf[0]/mpi_size;
    offset[0] = mpi_rank * count[0];
    memspace = H5Screate_simple(1, count, NULL);


    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);

    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                     plist_id, lambda0);*/




     /* Write lambda */

    dimsf[0] = nMP*mpi_size;
    filespace = H5Screate_simple(1, dimsf, NULL);

    dset_id = H5Dcreate(group_id, "lambL", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    count[0] = dimsf[0]/mpi_size;
    offset[0] = mpi_rank * count[0];
    memspace = H5Screate_simple(1, count, NULL);


    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);

    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, lambda);




   
    /* Write sprt */

    dimsf[0] = nboot;
    dimsf[1] = n;
    filespace = H5Screate_simple(2, dimsf, NULL);

    dset_id = H5Dcreate(group_id, "sprt", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace);

    count[0] = dimsf[0]/mpi_size;
    count[1] = dimsf[1];
    offset[0] = mpi_rank * count[0];
    offset[1] = 0;
    memspace = H5Screate_simple(2, count, NULL);


    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);

    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, sprt_in);


    H5Pclose(plist_id);
    H5Sclose(filespace);
    H5Sclose(memspace);
    H5Dclose(dset_id);
    H5Gclose(group_id);

}


void 
write2D (int DIMS1, int DIMS2, float *output, MPI_Comm comm)
{
	
	hid_t       file_id, dset_id, group_id, attribute_id;         /* file and dataset identifiers */
        hid_t       filespace, memspace;      /* file and memory dataspace identifiers */
        hsize_t     dimsf[2], dims[1];                 /* dataset dimensions */
        int         data[1];                    /* pointer to data buffer to write */
        hsize_t     count[2], counts[1];              /* hyperslab selection parameters */
        hsize_t     offset[2], offsets[1];
        hid_t       plist_id;                 /* property list identifier */
        int         i;
        herr_t      status;


        int mpi_size, mpi_rank;
        MPI_Info info  = MPI_INFO_NULL;
        MPI_Comm_size(comm, &mpi_size);
        MPI_Comm_rank(comm, &mpi_rank);	
	
	 /* Write support_*/
        dimsf[0] = DIMS1;
        dimsf[1] = DIMS2;
        filespace = H5Screate_simple(2, dimsf, NULL);

        dset_id = H5Dcreate(group_id, "support_", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Sclose(filespace);

        count[0] = dimsf[0]/mpi_size;
        count[1] = dimsf[1];
        offset[0] = mpi_rank * count[0];
        offset[1] = 0;
        memspace = H5Screate_simple(2, count, NULL);


        filespace = H5Dget_space(dset_id);
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);

        plist_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

        status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, output);
}

void
write_results(int bootstraps, int n_lambda, int n_features, float* b_hat, float* bic_scores, char OutFile[], MPI_Comm comm)
{

	hid_t       file_id, dset_id, group_id, attribute_id;         /* file and dataset identifiers */
    	hid_t       filespace, memspace;      /* file and memory dataspace identifiers */
    	hsize_t     dimsf[2], dims[1];                 /* dataset dimensions */
    	int         data[1];                    /* pointer to data buffer to write */
    	hsize_t     count[2], counts[1];              /* hyperslab selection parameters */
    	hsize_t     offset[2], offsets[1];
    	hid_t       plist_id;                 /* property list identifier */
    	int         i;
    	herr_t      status;


    	int mpi_size, mpi_rank;
    	MPI_Info info  = MPI_INFO_NULL;
    	MPI_Comm_size(comm, &mpi_size);
    	MPI_Comm_rank(comm, &mpi_rank);

    	plist_id = H5Pcreate(H5P_FILE_ACCESS);
    	H5Pset_fapl_mpio(plist_id, comm, info);
    	file_id = H5Fcreate(OutFile, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
	group_id = H5Gcreate(file_id, "Results", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    	H5Pclose(plist_id);
	
	/*Write Coeff_ */

	/*dimsf[0] = n_features;
	dimsf[1]  = 1;

    	filespace = H5Screate_simple(2, dimsf, NULL);;
    	dset_id = H5Dcreate(group_id, "coef_", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    	H5Sclose(filespace);

    	count[0] = dimsf[0]/mpi_size;
	count[1] = dimsf[1];
   	offset[0] = mpi_rank * count[0];
	offset[1] = 0;
    	memspace = H5Screate_simple(2, count, NULL);

    	filespace = H5Dget_space(dset_id);
    	H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);

	plist_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    	status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, b_hat);*/
	

	/*Write bic_scores*/ 

	dimsf[0] = bootstraps;
        dimsf[1] = n_lambda;
        filespace = H5Screate_simple(2, dimsf, NULL);

        dset_id = H5Dcreate(group_id, "scores_", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Sclose(filespace);

        count[0] = dimsf[0]/mpi_size;
        count[1] = dimsf[1];
        offset[0] = mpi_rank * count[0];
        offset[1] = 0;
        memspace = H5Screate_simple(2, count, NULL);


        filespace = H5Dget_space(dset_id);
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);

	plist_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

        status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, bic_scores);



	/* Write support_*/
	/*dimsf[0] = n_lambda;
    	dimsf[1] = n_features;
    	filespace = H5Screate_simple(2, dimsf, NULL);

    	dset_id = H5Dcreate(group_id, "support_", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    	H5Sclose(filespace);

    	count[0] = dimsf[0]/mpi_size;
    	count[1] = dimsf[1];
    	offset[0] = mpi_rank * count[0];
    	offset[1] = 0;
    	memspace = H5Screate_simple(2, count, NULL);


    	filespace = H5Dget_space(dset_id);
    	H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);

    	status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, support);*/
	
	dims[0] = n_features;
        //dimsf[1]  = 1;

        filespace = H5Screate_simple(1, dims, NULL);;
        dset_id = H5Dcreate(group_id, "coef_", H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Sclose(filespace);

        counts[0] = dims[0]/mpi_size;
        //count[1] = dimsf[1];
        offsets[0] = mpi_rank * counts[0];
        //offset[1] = 0;
        memspace = H5Screate_simple(1, counts, NULL);

        filespace = H5Dget_space(dset_id);
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets, NULL, counts, NULL);

        //plist_id = H5Pcreate(H5P_DATASET_XFER);
        //H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

        status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, b_hat); 

	H5Pclose(plist_id);
    	H5Sclose(filespace);
   	H5Sclose(memspace);
    	H5Dclose(dset_id);
    	H5Gclose(group_id);

}

void 
write_out (int rows, int cols, float *final_result, char OutFile[], MPI_Comm comm, char dataname[])
{
	 hid_t       file_id, dset_id, group_id, attribute_id;         /* file and dataset identifiers */
        hid_t       filespace, memspace;      /* file and memory dataspace identifiers */
        hsize_t     dimsf[2], dims[1];                 /* dataset dimensions */
        int         data[1];                    /* pointer to data buffer to write */
        hsize_t     count[2], counts[1];              /* hyperslab selection parameters */
        hsize_t     offset[2], offsets[1];
        hid_t       plist_id;                 /* property list identifier */
        int         i;
        herr_t      status;


        int mpi_size, mpi_rank;
        MPI_Info info  = MPI_INFO_NULL;
        MPI_Comm_size(comm, &mpi_size);
        MPI_Comm_rank(comm, &mpi_rank);

        plist_id = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(plist_id, comm, info);
        file_id = H5Fcreate(OutFile, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
        group_id = H5Gcreate(file_id, "Results", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Pclose(plist_id);
	
	dimsf[0] = rows;
        dimsf[1] = cols;
        filespace = H5Screate_simple(2, dimsf, NULL);

        dset_id = H5Dcreate(group_id, dataname, H5T_NATIVE_FLOAT, filespace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Sclose(filespace);

        count[0] = dimsf[0]/mpi_size;
        count[1] = dimsf[1];
        offset[0] = mpi_rank * count[0];
        offset[1] = 0;
        memspace = H5Screate_simple(2, count, NULL);


        filespace = H5Dget_space(dset_id);
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);

        plist_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

        status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                      plist_id, final_result);
	
	H5Pclose(plist_id);
        H5Sclose(filespace);
        H5Sclose(memspace);
        H5Dclose(dset_id);
        H5Gclose(group_id);
	H5Fclose(file_id);


}
