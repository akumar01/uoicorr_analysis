import sys, os
import pdb
import glob
import numpy as np
import argparse
from job_utils.idxpckl import Indexed_Pickle

###### Command line arguments #######
parser = argparse.ArgumentParser()

parser.add_argument('path')
args = parser.parse_args()
path = args.path

# Go through the provided directory, open up all the .dat files, and collect 
# the unique cov_params

dat_files = glob.globa('%s/*.dat' % path)

cov_params = []

for dat_file in dat_files:

    f = Indexed_Pickle(dat_file)
    f.init_read()

    total_tasks = len(f.index)
    n_features = f.header['n_features']

    for i in range(total_tasks):

        params = f.read(i)

        cp_ = params['cov_params']

        if cp_ not in cov_params:
            cov_params.append(cp_)


