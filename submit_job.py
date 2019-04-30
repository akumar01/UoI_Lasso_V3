import numpy as np
import h5py
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('data_path')
parser.add_argument('idx', type=int)
parser.add_argument('--name', default = 'test')
parser.add_argument('-r', '--run', action= 'store_true')
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

# Edit where to look for the data 
input_line_loc = [i for i in np.arange(len(fcontents)) if 'input=' in fcontents[i]]
input_line = 'input=%s' % new_data_path
fcontents[input_line_loc[0]] = input_line


# Edit the output coef and score files
coef_output_loc = [i for i in np.arange(len(fcontents) if 'cfile=' in fcontents[i])]
coef_output = 'cfile=%s'
fcontents[coef_output_loc[0]] = coef_output

score_output_loc = [i for i in np.arange(len(fcontents) if 'sfile=' in fcontents[i])]
score_output = 'sfile=%s'
fcontents[score_output_loc[0]] = score_output

with open('sbatch_%s' % name, 'w') as f:   
	f.write(fcontents)

if args.run
    os.system('chmod u+x %s' % run_file)
    # Submit the job
    os.system('sbatch %s ' % run_file)
