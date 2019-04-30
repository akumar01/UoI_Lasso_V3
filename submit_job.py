import numpy as np
import h5py
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('data_path')
parser.add_argument('idx', type=int)
parser.add_argument('--name', default = 'test')
args = parser.parse_args()

# Load the z-scored data
dat = np.load(args)
dat = dat['arr0']

# Separate into response (idx) and predictors
y = dat[:, idx]
X = np.delete(dat, idx, axis=1)

# Save as hdf5 file in the same folder as the datafile
dirpath, fpath = os.path.split(args.data_path)

new_data_path = '%s/data%d.h5' % (dirpath, args.idx)

with open('%s/data%d.h5' % (dirpath, args.idx), 'w') as f:
    f['X'] = X
    f['y'] = y


# Re-format the job script to point to the write data file, 
# and store the results of the fits to the right location
with open('job', 'r') as f:
    fcontents = f.readlines()

# Edit the where to look for the data 
input_line_loc = [i for i in np.arange(len(fcontents)) if 'input=' in fcontents[i]]
input_line = 'input=%s' % 

with open('sbatch_%s' % name, 'w') as f:
    