TARGETS=uoi
OBJS = distribute-data.o lasso.o bins.o manage-data.o CommandLineOptions.o


CXX=CC #for PC mpic++
#for PC EIGEN3=-I /usr/local/include/eigen3/
ADD_FLAGS = #-DCIRCULARDEPENDENCE #additional flags can be added here. 

BOOST = -I${BOOST_ROOT}/include
CXXFLAGS=-Wall -g  -O3 -std=c++11 $(EIGEN3) $(BOOST) $(GSL) -xMIC-AVX512 -qopenmp -dynamic -debug inline-debug-info -mkl -std=c++11  -fp-model fast=2
CC=CC #for PC mpicc
CCFLAGS=-g  -O3  -xMIC-AVX512 
#BOOSTFLAGS =  -L/usr/common/software/boost/1.63/intel/mic-knl/lib/ #-lboost_program_options -lboost_filesystem -lboost_system

BOOSTFLAGS =  -L${BOOST_ROOT}/lib/

.PHONY: all clean

all: $(TARGETS)

uoi : $(OBJS)
	$(CXX)  -o $@ $(OBJS)  $(CXXFLAGS) $(ADD_FLAGS) $(BOOSTFLAGS) -lhdf5 -lboost_program_options -lboost_filesystem -lboost_system -lgsl

%.o : %.cpp
	$(CXX) -c $<  $(CXXFLAGS) $(ADD_FLAGS) $(BOOSTFLAGS) -lboost_program_options -lboost_filesystem -lboost_system -lgsl 

%.o : %.c
	$(CC) -c  $< -lhdf5 $(CCFLAGS)

clean:
	rm -f $(TARGETS) $(OBJS)
