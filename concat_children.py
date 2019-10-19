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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('rootdir')
    args = parser.parse_args()
    comm = MPI.COMM_WORLD

    concatenate_children(comm, args.rootdir)

