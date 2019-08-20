import os, glob, sys
import json
import h5py
import pickle
import numpy as np
import pandas as pd
import itertools
import importlib
import struct
import pdb
import time
from job_manager import grab_files
import awkward as awk


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
def postprocess_v2(data_file, param_file, fields = None):
    
    data_list = []
    
    # Indexed pickle file
    param_file = Indexed_Pickle(param_file)
    
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
            data_dict['selection_method'] = selection_method 
            # Save memory
            data_dict['sigma'] = []
            if fields is None:
                for key in data_file[selection_method].keys():
                    try:
                        data_dict[key] = data_file[selection_method][key][i]
                    except:
                        pdb.set_trace()
            else:
                for key in fields:
                    if key in data_file[selection_method].keys():
                        data_dict[key] = data_file[selection_method][key][i]
            data_list.append(data_dict)
    return data_list

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
def postprocess_dir(jobdir, exp_type = None, fields = None, old_format = False, awkward=False):
    # Collect all .h5 files
    data_files = grab_files(jobdir, '*.dat', exp_type)
    print(len(data_files))
    # List to store all data
    data_list = []
    for i, data_file in enumerate(data_files):
        _, fname = os.path.split(data_file)
        jobno = fname.split('.dat')[0].split('job')[1]
        with open('%s/master/params%s.dat' % (jobdir, jobno), 'rb') as f2:
            if awkward:
                with open(data_file, 'rb') as f1:
                    f1 = pickle.load(f1)
                    d = postprocess_awkward(f1, f2)
            else:
                with h5py.File(data_file, 'r') as f1:
                    d = postprocess_v2(f1, f2, fields)

            data_list.extend(d)        
        print(i)
        
    # Copy to dataframe
    dataframe = pd.DataFrame(data_list)

    print(dataframe.shape)
    return dataframe
