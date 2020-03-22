import os, glob, sys
import h5py
import h5py_wrapper
import pickle
import numpy as np
import pandas as pd
import struct
import pdb
import time
import resource
import subprocess
# import awkward as awk
import sqlalchemy
import argparse
from schwimmbad import MPIPool
from mpi4py import MPI

from mpi_utils.ndarray import Gatherv_rows
from job_utils.results import ResultsManager
from job_utils.idxpckl import Indexed_Pickle
from postprocess_utils import grab_files

# Cantor pairing function returns a unique integer from 2 intermediate values
def cantor_pair(k1, k2):

    return int(1/2 * (k1 + k2)(k1 + k2 + 1) + k2)

class StreamWorker():

    def __init__(self, savename, save_beta):

        self.savename = savename
        self.save_beta = save_beta

    # On the master process, save to file as the results come in in the appropriate locations
    def stream(self, result):
        # Unpack the result and convert
        d = result[0]
        b = result[1]
        bhat = result[2]
        uids = result[3]

        b = np.array(b)
        bhat = np.array(bhat)
        uids = np.array(uids)

        assert(len(d) == b.shape[0])
        assert(b.shape[0] == bhat.shape[0])

        if b.shape[0] == 0:
            return

        if self.save_beta:
            # Initialize file objects if they have not yet been
            if not hasattr(self, 'beta_obj'):

                f1 = h5py.File('%s_beta.h5' % self.savename, 'w')
                try:
                    beta_table = f1.create_dataset('beta', b.shape, maxshape=(None, b.shape[1]))
                except:
                    print(b.shape)
                    pdb.set_trace()

                idxs_table = f1.create_dataset('ids', (b.shape[0],), maxshape=(None,))
                beta_hat_table = f1.create_dataset('beta_hat', bhat.shape, maxshape=(None, bhat.shape[1]))

                self.beta_obj = {}
                self.beta_obj['fobj'] = f1
                self.beta_obj['beta_table'] = beta_table
                self.beta_obj['beta_hat_table'] = beta_hat_table
                self.beta_obj['idxs_table'] = idxs_table

                # Append the beta
                self.beta_obj['beta_table'][:] = b
                self.beta_obj['beta_hat_table'][:] = bhat
                self.beta_obj['idxs_table'][:] = uids

            # Need to extend the dataframes
            else:

                shape = (self.beta_obj['beta_table'].shape[0] + b.shape[0], self.beta_obj['beta_table'].shape[1])
                self.beta_obj['beta_table'].resize(shape)
                self.beta_obj['beta_table'][-b.shape[0]:, :] = b

                self.beta_obj['idxs'].reshape((shape[0],))
                self.beta_obj['idxs'][-b.shape[0]:] = uids

                shape = (self.beta_obj['beta_hat_table'].shape[0] + bhat.shape[0], self.beta_obj['beta_hat_table'].shape[1])
                self.beta_obj['beta_hat_table'].resize(shape)
                self.beta_obj['beta_hat_table'][-bhat.shape[0]:, :] = bhat

        if not hasattr(self, 'data_obj'):

            self.data_obj = {}
            self.data_obj['fobj'] = Indexed_Pickle('%s_df.h5' % self.savename)
            self.data_obj['fobj'].init_save()

        self.data_obj['fobj'].save(d)

        # Log the memory usage
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print('Streamer, using %f memory' % mem)

    def close(self):

        if hasattr(self, 'beta_obj'):

            self.beta_obj['fobj'].close()

        if hasattr(self, 'data_obj'):

            self.data_obj['fobj'].close_save()

class PostprocessWorker():

    def __init__(self, jobdir, fields,
                 rank=0, size=0, burst=False,
                 buffer_loc=None):

        self.jobdir = jobdir
        self.fields = fields
        self.rank = rank
        self.size = size
        self.beta_list = []
        self.beta_hat_list = []
        self.data_list = []

        # Burst buffer usage
        self.burst = burst
        self.buffer_loc = buffer_loc


    def __call__(self, data_file):
        _, fname = os.path.split(data_file)
        t0 = time.time()
        jobno = fname.split('.dat')[0].split('_')[-1]
        # If burst, copy the data file to the burst first
        if self.burst:
            subprocess.Popen(['time', 'cp', data_file, self.buffer_loc], stdout=sys.stdout).communicate()
            # Change the data file path accordingly
            print('oi!')
            data_file = os.path.join(self.buffer_loc, fname)
        f1 = h5py_wrapper.load(data_file)
        f2 = '%s/master/params%s.dat' % (self.jobdir, jobno)

        d, b, bhat, uids = postprocess(f1, f2, jobno, self.fields)
        # Log the memory usage
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print('Rank %d, using %f memory' % (self.rank, mem))
        # Delete the data file from the burst
        if self.burst:
            subprocess.Popen(['rm', data_file])
        print('Run time: %f' % (time.time() - t0)) 
        return (d, b, bhat, uids)

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

# New format with results from multiple selection methods
def postprocess(data_file, param_file, jobno, fields = None):

    data_list = []
    beta_list = []
    beta_hat_list = []
    unique_ids = []

    # Indexed pickle file
    param_file = Indexed_Pickle(param_file)
    param_file.init_read()
    # print(len(param_file.index))
    for i in np.arange(len(param_file.index)):
        params = param_file.read(i)

        # Enumerate over selection methods and save a separate pandas row for each selection method
        selection_methods = list(data_file.keys())
        for j, selection_method in enumerate(selection_methods):

            # Use 2 level Cantor pairing function to map the file number, index, and selection method 
            # to a unique index 
            unique_id = cantor_pair(j, cantor_pair(jobno, i))
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
            data_dict['unique_id'] = unique_id
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
            unique_ids.append(unique_id)
            # this guy is unsparsified
            del data_dict['beta']

            data_list.append(data_dict)

    param_file.close_read()

    beta_list = np.array(beta_list)
    beta_hat_list = np.array(beta_hat_list)

    return data_list, beta_list, beta_hat_list, unique_ids

# New postprocessing function that should subsume all desired functionality
# Can be run in parallel, if desired

# For nonstandard runs, just need to specify which param file and which indices 
# the data file corresponds to (unimplementec) 
def postprocess_run(jobdir, savename, exp_type, fields, save_beta=False,
                    comm=None, return_dframe=True, burst=False, buffer_loc=None):

   # Distribute postprocessing across ranks, if desired
    if comm is not None:
        # Collect all .h5 files
        if comm.rank == 0:
            print(jobdir)
            print(exp_type)
            data_files = glob.glob(jobdir + '/%s/*.dat' % exp_type)
            #data_files = grab_files(jobdir, '*.dat', exp_type)
            data_files = [(d) for d in data_files]
            pdb.set_trace()
            print(len(data_files))
        else:
            data_files = None

        rank = comm.rank
        size = comm.size
        # print('Rank %d' % rank)
        master = StreamWorker(savename, save_beta)
        worker = PostprocessWorker(jobdir, fields, rank, size, burst=burst, buffer_loc=buffer_loc)
        pool = MPIPool(comm)
        pool.map(worker, data_files, callback=master.stream, track_results=False)
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        pool.close()

        if rank == 0:
            master.close()

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

    # gc.set_debug(gc.DEBUG_LEAK)

    parser = argparse.ArgumentParser()
    parser.add_argument('jobdir')
    parser.add_argument('savename')
    parser.add_argument('exp_type')
    parser.add_argument('--nfiles', type=int, default = None)
    parser.add_argument('--save_beta', action='store_true')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--burst', action='store_true')
    parser.add_argument('--buffer_loc', default=None)
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
                    return_dframe=False, comm=comm, burst=args.burst, buffer_loc=args.buffer_loc)
    
