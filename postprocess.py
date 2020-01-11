import os, glob, sys
import h5py
import h5py_wrapper
import pickle
import numpy as np
import pandas as pd
import struct
import pdb
import time
# import awkward as awk
import sqlalchemy
import argparse
from schwimmbad import MPIPool
from mpi4py import MPI

from mpi_utils.ndarray import Gatherv_rows
from job_utils.results import ResultsManager
from job_utils.idxpckl import Indexed_Pickle
from postprocess_utils import grab_files

class PostprocessWorker():

    def __init__(self, jobdir, fields, savename,
                 rank=0, size=0):

        self.jobdir = jobdir
        self.fields = fields
        self.rank = rank
        self.size = size

        self.savename = savename

        self.beta_list = []
        self.beta_hat_list = []
        self.data_list = []


    def __call__(self, data_file):
        _, fname = os.path.split(data_file)
        print('Oi!')
        jobno = fname.split('.dat')[0].split('_')[-1]
        with h5py.File(data_file, 'r') as f1:
            f2 = '%s/master/params%s.dat' % (self.jobdir, jobno)
            d, b, bhat = postprocess(f1, f2, self.fields)

        return (d, b, bhat)

    # On the master process, save to file as the results come in in the appropriate locations
    def stream(self, result):
        # Unpack the result and convert
        d = result[0]
        b = result[1]
        bhat = result[2]

        b = np.array(b)
        bhat = np.array(bhat)
        dframe = pd.DataFrame(d)

        # Initialize file objects if they have not yet been
        if not hasattr(self, 'beta_obj'):

            f1 = h5py.File('%s_beta.h5' % self.savename, 'w')
            beta_table = f.create_dataset('beta', b.shape, maxshape=(None, b.shape[1]))
            beta_hat_table = f.create_dataset('beta_hat', bhat.shape, maxshape=(None, bhat.shape[1]))

            self.beta_obj = {}
            self.beta_obj['fobj'] = f1
            self.beta_obj['beta_table'] = beta_table
            self.beta_obj['beta_hat_table'] = beta_hat_table

            # Append the beta
            self.beta_obj['beta_table'][:] = b
            self.beta_obj['beta_hat_table'][:] = bhat

        # Need to extend the dataframes
        else:

            shape = (self.beta_obj['beta_table'].shape[0] + b.shape[0], self.beta_obj['beta_table'].shape[1])
            self.beta_obj['beta_table'].resize(shape)
            self.beta_obj['beta_table'][-b.shape[0]:, :] = b

            shape = (self.beta_obj['beta_hat_table'].shape[0] + bhat.shape[0], self.beta_obj['beta_hat_table'].shape[1])
            self.beta_obj['beta_hat_table'].resize(shape)
            self.beta_obj['beta_hat_table'][-bhat.shape[0]:, :] = bhat

        if not hasattr(self, 'data_obj'):

            self.data_obj = {}s
            self.data_obj['path'] = '%s_df.h5' % savename

        dframe.to_hdf(self.data_obj['path'], 'dframe_table', append=True)

        del d
        del b
        del bhat

    def extend(self, result):

        # Unpack the result and extend the relevant variables
        d = result[0]
        b = result[1]
        bhat = result[2]

        self.beta_list.extend(b)
        self.beta_hat_list.extend(bhat)
        self.data_list.extend(d)

        print(len(self.beta_list))

    def save(self, save_beta=False):

        self.beta_list = np.array(self.beta_list)
        self.beta_hat_list  = np.array(self.beta_hat_list)

        # Save the dataframe to sql, beta and beta_hat to hdf5
        f = h5py.File('%s_beta.h5' % self.savename, 'w')
        beta_table = f.create_dataset('beta', self.beta_list.shape)
        beta_hat_table = f.create_dataset('beta_hat', self.beta_hat_list.shape)

        beta_table[:] = self.beta_list
        beta_hat_table[:] = self.beta_hat_list

        f.close()
        dataframe = pd.DataFrame(self.data_list)
        dataframe.to_pickle('%s_df.dat' % self.savename)

        return dataframe

    def close(self):

        if hasattr(self, 'beta_obj'):

            self.beta_obj['fobj'].close()


# New format with results from multiple selection methods
def postprocess(data_file, param_file, fields = None):

    data_list = []
    beta_list = []
    beta_hat_list = []

    # Indexed pickle file
    param_file = Indexed_Pickle(param_file)
    param_file.init_read()
    print(len(param_file.index))
    for i in range(len(param_file.index)):
        params = param_file.read(i)
        # Enumerate over selection methods and save a separate pandas row for each selection method
        selection_methods = list(data_file.keys())
        for selection_method in selection_methods:
            data_dict = params.copy()
            # Remove refernces to all selection_methods and the fields to save for those
            # selection methods
            del data_dict['selection_methods']
            del data_dict['fields']

            # Remove things we will never refer to, and will cause later problems for serialization
            del data_dict['stability_selection']
            del data_dict['gamma']
            del data_dict['l1_ratios']
            del data_dict['sub_iter_params']
            data_dict['selection_method'] = selection_method

            for key in fields:
                if key in data_file[selection_method].keys():
                    data_dict[key] = data_file[selection_method][key][i][0]

            # Flatten dictionaries associated with betadict and cov_params
            for key in ['betadict']:

                for subkey, value in data_dict[key].items():

                    data_dict[subkey] = value

                del data_dict[key]

            beta_list.append(data_file[selection_method]['beta'][i].ravel())
            beta_hat_list.append(data_file[selection_method]['beta_hats'][i].ravel())

            # this guy is unsparsified
            del data_dict['beta']

            data_list.append(data_dict)
    param_file.close_read()

    beta_list = np.array(beta_list)
    beta_hat_list = np.array(beta_hat_list)

    return data_list, beta_list, beta_hat_list

# New postprocessing function that should subsume all desired functionality
# Can be run in parallel, if desired

# For nonstandard runs, just need to specify which param file and which indices 
# the data file corresponds to (unimplementec) 
def postprocess_run(jobdir, savename, exp_type, fields, save_beta=False, n_features=500,
                    comm=None, return_dframe=True):

    # Collect all .h5 files
    data_files = grab_files(jobdir, '*.dat', exp_type)
    data_files = [(d) for d in data_files]
    # Distribute postprocessing across ranks, if desired
    if comm is not None:
        print(len(data_files))
        rank = comm.rank
        size = comm.size
        worker = PostprocessWorker(jobdir, fields, savename, rank, size)
        pool = MPIPool(comm)
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        pool.map(worker, data_files, callback=worker.stream)
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        pool.close()

        if rank == 0:
            worker.close()

    else:
        worker = PostprocessWorker(jobdir, fields, savename)
        for i, data_file in enumerate(data_files):
            t0 = time.time()
            result = worker(data_file)
            worker.extend(result)
            print(time.time() - t0)
        dframe = worker.save(save_beta)

    if return_dframe:
        return dframe

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('jobdir')
    parser.add_argument('savename')
    parser.add_argument('exp_type')
    parser.add_argument('n_features')
    parser.add_argument('--nfiles', type=int, default = None)
    parser.add_argument('--save_beta', action='store_true')
    parser.add_argument('--parallel', action='store_true')
    args = parser.parse_args()

    # Fix the fields to be everything we are intersted in
    fields = ['sa', 'FNR', 'FPR', 'ee', 'r2', 'MSE', 'ss']

    # Create a comm world object
    if args.parallel:
        comm = MPI.COMM_WORLD
    else:
        comm = None
    # postprocess_parallel(comm, args.jobdir, args.savename, args.exp_type, fields,
    #                      args.save_beta, args.n_features, args.nfiles)

    postprocess_run(args.jobdir, args.savename, args.exp_type, fields, args.save_beta,
                    args.n_features, return_dframe=False, comm=comm)
    
