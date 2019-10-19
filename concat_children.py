import pdb
import time
import numpy as np
import pickle
import h5py_wrapper
from job_utils.results import ResultsManager
import argparse
import os
import glob
from mpi4py import MPI
import pandas as pd
import sqlalchemy

# Slightly different than ResultsManager.concatenate because we have
# non-contiguous sets of indices, but want no gaps between them
def concatenate_children(comm, root_dir):

    # Assemble the list of subdirectories that need to be processed
    dirlist = []
    for root, dirs, files in os.walk(root_dir):
        for d in dirs:
            p = os.path.join(root, d)
            if 'node' in p:
                dirno = p.split('dir')[1].split('/')[0]
                nodeno = p.split('node')[1]
                if len(glob.glob('%s/master_%s_%s.dat' % (p, dirno, nodeno))) == 0:
                    dirlist.append(p)

    # Chunk the dirlist
    chunk_dirlist = np.array_split(dirlist, comm.size)
    rank_dirlist = chunk_dirlist[comm.rank]
    print(len(rank_dirlist))
    for i, p in enumerate(rank_dirlist):

        t0 = time.time()
        rmanager = ResultsManager.restore_from_directory(p)
        master_list = []
        dirno = p.split('dir')[1].split('/')[0]
        nodeno = p.split('node')[1]

        for i, child in enumerate(rmanager.children):

            try:
                child_data = h5py_wrapper.load(child['path'])
                child_data['idx'] = child['idx']
                master_list.append(child_data)
            except:
                continue

        # Pickle away
        with open('%s/master_%s_%s.dat' % (p, dirno, nodeno), 'wb') as f:
            f.write(pickle.dumps(master_list))
        print('Task %d/%d, %f s' % (i + 1, len(dirlist), time.time()- t0))

# Need to iterate through all the subfolders and create a lookup table for the param file
# paths for that each node relied on
def grab_node_params(root_dir):

    node_lookup_table = []

    for root, dirs, files in os.walk(root_dir):
        for d in dirs:
            p = os.path.join(root, d)
            if 'node' in p:
                dirno = int(p.split('dir')[1].split('/')[0])
                nodeno = int(p.split('node')[1])
                with open('%s/node_param_file.pkl', 'rb') as f:
                    node_idx_list = pickle.load(f)
                table_row = {}
                table_row['dir'] = dirno
                table_row['node'] = nodeno
                table_row['param_file'] = list(node_idx_list.keys())[0]

                node_lookup_table.append(table_row)
    # Convert to pandas dataframe
    node_lookup_table = pd.DataFrame(node_lookup_table)
    # Save
    with open('%s/node_lookup_table.dat' % root_dir, 'wb') as f:
        f.write(pickle.dumps(node_lookup_table))

# Take the outputs of postprocessing for the 2 UoI folders and concatenate them
def merge_sfn_uoi(uoi1_path, uoi2_path, savepath, df_=True, beta_=True, bhat_=True):

    # First handle the sql tables
    if df_:
        # Establish db connections
        uoi1_engine = sqlalchemy.create_engine('sqlite:///%s/uoi1.db')
        uoi1_con = lasso_engine.connect()

        uoi2_engine = sqlalchemy.create_engine('sqlite:///%s/uoi2.db')
        uoi2_con = lasso_engine.connect()

        # Load dataframes
        uoi1_df = pd.read_sql_table('pp_df', uoi1_con)
        uoi2_df = pd.read_sql_table('pp_df', uoi2_con)

        # Concatenate them
        uoi_df = pd.concat([uoi1_df, uoi2_df])

        # Save
        sql_engine = sqlalchemy.create_engine('sqlite:///%s/uoi.db' % savepath, echo=False)
        t0 = time.time()
        uoi_df.to_sql('pp_df', sql_engine, if_exists='replace')
        print('sql write time: %f' % (time.time() - t0))

    # Now beta
    if beta_:
        # Load
        beta1 = h5py.File('%s/uoi1_beta.h5' % uoi1_path, 'r')
        beta2 = h5py.File('%s/uoi2_beta.h5' % uoi2_path, 'r')

        # Concatenate
        beta1_array = beta1['beta'][:]
        beta2_array = beta2['beta'][:]

        beta = np.vstack((beta1_array, beta2_array))

        del beta1_array
        del beta2_array

        # Save
        t0 = time.time()
        f1 = h5py.File('%s/uoi_beta.h5' % savename, 'w')
        beta_table = f1.create_dataset('beta', beta.shape)
        beta_table[:] = beta
        f1.close()
        print('beta write time: %f' % (time.time() - t0))

        del beta

    # Repeat process for beta_hat
    if bhat_:
        # Load
        beta1 = h5py.File('%s/uoi1_beta_hat.h5' % uoi1_path, 'r')
        beta2 = h5py.File('%s/uoi2_beta_hat.h5' % uoi2_path, 'r')

        # Concatenate
        beta1_array = beta1['beta_hat'][:]
        beta2_array = beta2['beta_hat'][:]

        beta = np.vstack((beta1_array, beta2_array))

        del beta1_array
        del beta2_array

        # Save
        t0 = time.time()
        f1 = h5py.File('%s/uoi_beta_hat.h5' % savename, 'w')
        beta_table = f1.create_dataset('beta_hat', beta.shape)
        beta_table[:] = beta
        f1.close()
        print('beta write time: %f' % (time.time() - t0))

        del beta



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('rootdir')
    args = parser.parse_args()
    comm = MPI.COMM_WORLD

    concatenate_children(comm, args.rootdir)

