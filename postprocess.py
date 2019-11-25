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
from mpi4py import MPI

from mpi_utils.ndarray import Gatherv_rows
from job_utils.results import ResultsManager
# Eventually turn this into its own standalone storage solution
class Indexed_Pickle():

    def __init__(self, file):
        self.file = file
        file.seek(0, 0)
        index_loc = file.read(8)
        # Some weird Windows bullshit
        if os.name == 'nt':
            index_loc = struct.unpack('q', index_loc)[0]
        else:
            index_loc = struct.unpack('L', index_loc)[0]
        total_tasks = pickle.load(file)
        n_features = pickle.load(file)
        file.seek(index_loc, 0)
        self.index = pickle.load(file)
        self.index_length = total_tasks

    def read(self, idx):

        self.file.seek(self.index[idx], 0)
        data = pickle.load(self.file)
        return data

# Crawl through a subdirectory and grab all files contained in it matching the
# provided criteria:
def grab_files(root_dir, file_str, exp_type = None):
    run_files = []
    for root, dirs, files in os.walk(root_dir):
        for d in dirs:
            p = os.path.join(root, d)
            if exp_type is not None:
                if exp_type in p:
                    run_files.extend(glob.glob('%s/%s' % (p, file_str)))
            else:
                run_files.extend(glob.glob('%s/%s' % (p, file_str)))

    #run_files = natsort.natsorted(run_files)
    return run_files

# Common postprocessing operations on a single data file
def postprocess(data_file, param_file, fields = None):
    data_list = []

    # Indexed pickle file
    param_file = Indexed_Pickle(param_file)

    for i in range(param_file.index_length):
        params = param_file.read(i)
        data_dict = params.copy()
        # Do not store Sigma to save memory
        data_dict['sigma'] = []
        if fields is None:
            for key in data_file.keys():
                data_dict[key] = data_file[key][i]
        else:
            for key in fields:
                data_dict[key] = data_file[key][i]

        data_list.append(data_dict)

    return data_list

# New format with results from multiple selection methods
def postprocess_v2(data_file, param_file, idx, fields = None):

    data_list = []
    beta_list = []
    beta_hat_list = []

    # Indexed pickle file
    param_file = Indexed_Pickle(param_file)
    print(param_file.index_length)
    for i in range(param_file.index_length):
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

            if fields is None:
                for key in data_file[selection_method].keys():
                    try:
                        data_dict[key] = data_file[selection_method][key][i][0]
                    except:
                        pdb.set_trace()
            else:
                for key in fields:
                    if key in data_file[selection_method].keys():
                        data_dict[key] = data_file[selection_method][key][i][0]

            # Flatten dictionaries associated with betadict and cov_params
            for key in ['cov_params', 'betadict']:

                for subkey, value in data_dict[key].items():

                    data_dict[subkey] = value


                del data_dict[key]

            beta_list.append(data_dict['beta'].ravel())
            beta_hat_list.append(data_file[selection_method]['beta_hats'][i].ravel())

            del data_dict['beta']

            data_dict['indices'] = np.array([idx, i])
            data_list.append(data_dict)

    beta_list = np.array(beta_list)
    beta_hat_list = np.array(beta_hat_list)

    return data_list, beta_list, beta_hat_list

def postprocess_emergency(master_file, param_file, fields):

    data_list = []
    beta_list = []
    beta_hat_list = []

    # Indexed pickle file
    # Will likely need to modify the path here
    param_file = Indexed_Pickle(param_file)
    param_file.init_read()

    # Load up the master file
    with open(master_file, 'rb') as mf:

        children = pickle.load(master_file)

        for i, child in enumerate(children):

            child_index = child['idx']
            params = param_file.read(child_index)

            # Enumerate over selection methods and save a separate pandas row for each selection method
            selection_methods = list(child.keys())
            # Exclude idx

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

                if fields is None:
                    for key in child[selection_method].keys():
                        try:
                            data_dict[key] = child[selection_method][key][0]
                        except:
                            pdb.set_trace()
                else:
                    for key in fields:
                        if key in child[selection_method].keys():
                            data_dict[key] = child[selection_method][key][0]

                # Flatten dictionaries associated with betadict and cov_params
                for key in ['cov_params', 'betadict']:

                    for subkey, value in data_dict[key].items():

                        data_dict[subkey] = value


                    del data_dict[key]

                beta_list.append(data_dict['beta'].ravel())
                beta_hat_list.append(child[selection_method]['beta_hats'].ravel())

                del data_dict['beta']

                data_list.append(data_dict)

        beta_list = np.array(beta_list)
        beta_hat_list = np.array(beta_hat_list)

        param_file.close_read()
        pdb.set_trace()
        return data_list, beta_list, beta_hat_list


# Use when the results contain an awkward array
def postprocess_awkward(data_file, param_file):

    data_list = []

    # Indexed pickle file
    param_file = Indexed_Pickle(param_file)

    for i in range(param_file.index_length):
        params = param_file.read(i)
        data_dict = params.copy()

        # Remove refernces to all selection_methods and the fields to save for those
        # selection methods
        if 'selection_methods' in data_dict.keys():
            del data_dict['selection_methods']
        if 'fields' in data_dict.keys():
            del data_dict['fields']

        data_dict['sigma'] = []

        for key in data_file.columns:
            try:
                data_dict[key] = data_file[key][i]
            except:
                pdb.set_trace()

        data_list.append(data_dict)
    return data_list

# Postprocess an entire directory of data, will assume standard nomenclature of
# associated parameter files
# exp_type: only postprocess results for the given exp_type
# fields (list): only return data for the fields given in fields (useful for saving
# memory)
# old format: Use postprocess instead of postprocess_v2
# awkward: Is the data saved as an awkward array?
def postprocess_dir(jobdir, savename, exp_type, fields = None, old_format = False, awkward=False,
                    n_features=500):

    # Collect all .h5 files
    data_files = grab_files(jobdir, '*.dat', exp_type)
    print(len(data_files))
    # List to store all data
    data_list = []

    f = h5py.File('%s_beta.h5' % savename, 'w')
    sql_engine = sqlalchemy.create_engine('sqlite:///%s.db' % savename, echo=False)

    # Process the first file to initialize the datasets to store beta and beta_hat
    _, fname = os.path.split(data_files[0])
    jobno = fname.split('.dat')[0].split('job')[1]
    with open('%s/master/params%s.dat' % (jobdir, jobno), 'rb') as f2:
        if awkward:
            with open(data_files[0], 'rb') as f1:
                f1 = pickle.load(f1)
                d = postprocess_awkward(f1, f2)
        else:
            with h5py.File(data_files[0], 'r') as f1:
                d, b, bhat = postprocess_v2(f1, f2, jobno, fields)
                print(list(d[0].keys()))
        data_list.extend(d)

    beta_table = f.create_dataset('beta', (len(data_files) *  b.shape[0], n_features),
                                  maxshape = (None, n_features))
    beta_hat_table = f.create_dataset('beta_hat', (len(data_files) * bhat.shape[0], n_features),
                                      maxshape = (None, n_features))

    # Populate the datsets with the arrays from the first file
    beta_table[0:b.shape[0], :] = b
    beta_hat_table[0:bhat.shape[0], :] = bhat

    bidx = b.shape[0]
    bhatidx = bhat.shape[0]

    for i, data_file in enumerate(data_files[1:]):
        _, fname = os.path.split(data_file)

        jobno = fname.split('.dat')[0].split('job')[1]
        with open('%s/master/params%s.dat' % (jobdir, jobno), 'rb') as f2:
            if awkward:
                with open(data_file, 'rb') as f1:
                    f1 = pickle.load(f1)
                    d = postprocess_awkward(f1, f2)
            else:
                with h5py.File(data_file, 'r') as f1:
                    d, b, bhat = postprocess_v2(f1, f2, fields)

            data_list.extend(d)

        # Populate the datasets
        beta_table[bidx:b.shape[0], :] = b
        beta_hat_table[bhatidx:bhat.shape[0], :] = bhat

        bidx += b.shape[0]
        bhatidx += bhat.shape[0]

        del b
        del bhat
        print(i)

    # Copy to dataframe
    dataframe = pd.DataFrame(data_list)
    dataframe.to_sql('pp_df', sql_engine, if_exists='append')
    f.close()

    return dataframe


# Split the postprocessing across many threads. Run this on catscan so everything can
# be held in memory and then gathered at the end
def postprocess_parallel(comm, jobdir, savename, exp_type, fields,
                         save_beta=False, n_features=500, nfiles=None):

    rank = comm.rank
    size = comm.size

    # Collect all .h5 files
    data_files = grab_files(jobdir, '*.dat', exp_type)

    if nfiles is not None:
        data_files = data_files[:nfiles]

    if rank == 0:
        print(len(data_files))

    # Chunk the data files across ranks
    task_list = np.array_split(data_files, size)[rank]

    # List to store all data
    data_list = []
    beta_list = []
    beta_hat_list = []

    dummy_list = []

    for i, data_file in enumerate(task_list):
        t0 = time.time()
        _, fname = os.path.split(data_file)

        jobno = fname.split('.dat')[0].split('job')[1]
        with open('%s/master/params%s.dat' % (jobdir, jobno), 'rb') as f2:
            with h5py.File(data_file, 'r') as f1:
                d, b, bhat = postprocess_v2(f1, f2,  int(jobno), fields)
            data_list.extend(d)
            beta_list.extend(b)
            beta_hat_list.extend(bhat)

        dummy_indices = np.zeros((2, b.shape[0]))
        dummy_indices[0, :] = jobno
        dummy_indices[1, :] = np.arange(b.shape[0])
        dummy_list.extend(dummy_indices)

        print('Rank %d completed task %d/%d in %f s' % (comm.rank, i + 1, len(task_list), time.time() - t0))



    # Gather beta and beta hat as arrays
    beta_list = np.array(beta_list)
    beta_hat_list = np.array(beta_hat_list)

    # These arrays will likely be too large to gather in one go. Therefore, we will partition them so that
    # their size in bits on memory is less than 2^31 bit
    beta_list = np.array_split(beta_list, 10, axis = 0)
    beta_hat_list = np.array_split(beta_hat_list, 10, axis = 0)

    beta_list_gathered = []
    beta_hat_list_gathered = []
    t0 = time.time()
    for subarray in beta_list:
        beta_list_gathered.append(Gatherv_rows(send=subarray, comm=comm))
    if rank == 0:
        print('beta list gather time: %f' % (time.time() - t0))
    t0 = time.time()
    for subarray in beta_hat_list:
        beta_hat_list_gathered.append(Gatherv_rows(send=subarray, comm=comm))
    if rank == 0:
        print('beta hat list gather time: %f' % (time.time() - t0))

    print(len(data_list))

    # Gather the data list
    t0 = time.time()

    comm.gather(data_list, root=0)

    if rank == 0:
        print('data list gather time: %f' % (time.time() - t0))
        beta_list = np.array(beta_list_gathered)
        beta_hat_list = np.array(beta_hat_list_gathered)

        # Write to disk
        if save_beta:
            t0 = time.time()
            f1 = h5py.File('%s_beta.h5' % savename, 'w')
            f1.create_dataset('beta', beta_list.shape)
            beta_table[:] = beta_list
            f1.close()
            print('beta write time: %f' % (time.time() - t0))

        t0 = time.time()
        f2 = h5py.File('%s_beta_hat.h5' % savename, 'w')
        beta_table = f2.create_dataset('beta_hat', beta_hat_list.shape)
        beta_table[:] = beta_hat_list
        f2.close()
        print('beta hat write time: %f' % (time.time() - t0))

        sql_engine = sqlalchemy.create_engine('sqlite:///%s.db' % savename, echo=False)
        data_list = [elem in sublist for sublist in data_list for elem in sublist]
        data_list = pd.DataFrame(data_list)
        t0 = time.time()
        data_list.to_sql('pp_df', sql_engine, if_exists='replace')
        print('sql write time: %f' % (time.time() - t0))

def postprocess_emergency_dir(jobdir, savename, exp_type, save_beta=True,
                         n_features=500):

    fields = ['FNR', 'FPR', 'sa', 'R2', 'ee', 'MSE']

    data_list = []
    beta_list = []
    beta_hat_list = []

    master_files = glob.glob('%s/master*' % jobdir)

    with open('%s/node_lookup_table.dat' % jobdir, 'rb') as f:
        node_lookup_table = pickle.load(f)

    for master_file in master_files:

        # Get the dir and node numbers
        dirno = int(master_file.split('master_')[1].split('_')[0])
        nodeno = int(master_file.split('.dat')[0].split('_')[-1])
        param_file = node_lookup_table.loc[node_lookup_table['dirno'] == dirno
                                          && node_lookup_table['nodeno'] = nodeno].iloc[0]['param_file']

        d, b, bhat = postprocess_emergency(master_file, param_file, fields)
        data_list.extend(d)
        beta_list.extend(b)
        beta_hat_list.extend(bhat)


    # Write data list
    sql_engine = sqlalchemy.create_engine('sqlite:///%s.db' % savename, echo=False)
    data_list = pd.DataFrame(data_list)
    t0 = time.time()
    data_list.to_sql('pp_df', sql_engine, if_exists='replace')
    print('sql write time: %f' % (time.time() - t0))

    # Write beta to file
    if save_beta:

        beta_list = np.array(beta_list)
        beta_hat_list = np.array(beta_hat_list)

        t0 = time.time()
        f1 = h5py.File('%s_beta.h5' % savename, 'w')
        f1.create_dataset('beta', beta_list.shape)
        beta_table[:] = beta_list
        f1.close()
        print('beta write time: %f' % (time.time() - t0))

        t0 = time.time()
        f2 = h5py.File('%s_beta_hat.h5' % savename, 'w')
        beta_table = f2.create_dataset('beta_hat', beta_hat_list.shape)
        beta_table[:] = beta_hat_list
        f2.close()
        print('beta hat write time: %f' % (time.time() - t0))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('jobdir')
    parser.add_argument('savename')
    parser.add_argument('exp_type')
    parser.add_argument('n_features')
    parser.add_argument('--nfiles', type=int, default = None)
    parser.add_argument('--save_beta', action='store_true')
    args = parser.parse_args()

    # Fix the fields to be everything we are intersted in
    fields = ['sa', 'FNR', 'FPR', 'ee', 'r2', 'MSE']

    # Create a comm world object
    # comm = MPI.COMM_WORLD
    # postprocess_parallel(comm, args.jobdir, args.savename, args.exp_type, fields,
    #                      args.save_beta, args.n_features, args.nfiles)


    postprocess_emergency_dir(args.jobdir, args.savename, args.exp_type, args.save_beta,
                             args.n_features)
