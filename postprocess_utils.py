import os
import pdb
import glob
from collections import Counter
import numpy as np
import scipy.integrate as integrate
from matplotlib import colors
import natsort 

from misc import group_dictionaries, calc_avg_cov


# Crawl through a subdirectory and grab all files contained in it matching the
# provided criteria:
def grab_files(root_dir, file_str, exp_type = None):
    run_files = []
    if exp_type is not None:

        for root, dirs, files in os.walk(root_dir):
            if exp_type is not None:
                for d in dirs:
                    p = os.path.join(root, d)
                    if exp_type in p:
                        run_files.extend(glob.glob('%s/%s' % (p, file_str)))
    else:
        run_files.extend(glob.glob('%s/%s' % (root_dir, file_str)))

    run_files = natsort.natsorted(run_files)
    return run_files

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def filter_by_dict(df, root_key, dict_filter):

    col = df[root_key].values

    filtered_idxs = []

    for i, c in enumerate(col):
        match = True
        for key, val in dict_filter.items():
            if c[key] != val:
                match = False
        if match:
            filtered_idxs.append(i)

    return df.iloc[filtered_idxs]

# Shortcut to apply multiple filters to pandas dataframe
def apply_df_filters(dtfrm, invert=False, **kwargs):

    filtered_df = dtfrm

    for key, value in kwargs.items():

        # If the value is the dict
        if type(value) == dict:

            filtered_df = filter_by_dict(filtered_df, key, value)

        else:
            if type(value) == list:
                if invert:
                    filtered_df = filtered_df.loc[np.invert(filtered_df[key].isin(value))]
                else:
                    filtered_df = filtered_df.loc[filtered_df[key].isin(value)]
            else:
                if invert:
                    filtered_df = filtered_df.loc[filtered_df[key] != value]
                else:
                    filtered_df = filtered_df.loc[filtered_df[key] == value]

    return filtered_df

# Take a (nested) list of indices that reference .iloc and return the
# .index attribute that maps their location to the original unfiltered
# datraframe
def map_idxs(df, idxs):

    ridxs = []
    df_row_idxs = list(df.index)

    for idx in idxs:

        if not np.isscalar(idx):

            ridxs.append(map_idxs(df, idx))

        else:

            ridxs.append(df_row_idxs[idx])

    return ridxs

# Task: Get the unique dictionaries of cov_params from the dataframe
def unique_cov_params(df):
    cov_params = df['cov_params'].values
    unique_cov_params, cov_idxs = group_dictionaries(cov_params, None)
    unique_cov_dicts = []
    for ucp in unique_cov_params:
        ucd = {'correlation' : ucp[0], 'block_size' : ucp[1], 'L' : ucp[2], 't': ucp[3]}
        unique_cov_dicts.append(ucd)

    return unique_cov_dicts, cov_idxs

# Task: Given the indices of repeated elements, select the FNR and FPR and average them together
def average_fields(df, fields, rep_idxs, return_std=False):
    results = []
    results_std = []
    for i, field in enumerate(fields):
        # Pass in None to not average
        if rep_idxs is None:
            values = np.zeros(df.shape[0])
            for j in range(df.shape[0]):
                values[j] = df.iloc[j][field]
        else:
            values = np.zeros(len(rep_idxs))
            values_std = np.zeros(len(rep_idxs))
            for j, rep_idx in enumerate(rep_idxs):
                try:
                    values[j] = np.mean(df.iloc[rep_idx][field])
                except:
                    pdb.set_trace()
                values_std[j] = np.std(df.iloc[rep_idx][field])
        results.append(values)
        results_std.append(values_std)
    if return_std:
        return results, results_std
    else:
        return results

# Task: start from the smallest list/array, and then truncate the rest of
# def array_intersection(jagged_array):

#     # Take "intersections" between successive elements of the jagged array. keep track
#     # of the smallest common array.
#     ref = jagged_array[0]

#     for i in range(1, jagged_array.size):
#         intersection = np.array([elem for elem in jagged_array[i] if elem in ref])
#         if intersection.size < ref.size:
#             ref = intersection

#     ref = ref[np.newaxis, :]
#     # Tile the ref
#     rect_array = np.tile(ref, (jagged_array.size, 1))

#     # Keep track of the indices in each jagged element that survive through the intersection
#     # process
#     ref_idxs = []

#     for i in range(jagged_array.size):
#         ref_idxs.append(np.array([idx for idx in np.arange(jagged_array[i].size) if jagged_array[i][idx] in ref]))

#     pdb.set_trace()
#     return rect_array, ref_idxs

def array_intersection(jagged_array):

    ref = jagged_array[0]
    fluff = []

    for i in range(1, jagged_array.size):
        complement = [elem for elem in ref if elem not in jagged_array[i]]
        print('Size of complement: %d' % len(complement))
        fluff.extend([elem for elem in complement if elem not in fluff])

    pdb.set_trace()


def weighted_rates(df, n_features):

    sparsity = np.unique(df['sparsity'])

    weighted_rates = []
    avg_cov = []
    for i, s, in enumerate(sparsity):
        df_ = apply_df_filters(df, sparsity = s)

        # Calculate average covariance
        cov_params, rep_idxs = unique_cov_params(df_)
        avg_cov.append(np.array([calc_avg_cov(n_features, **cp) for cp in cov_params]))
        FNR_, FPR_ = average_fields(df_, ['FNR', 'FPR'], rep_idxs)
        # Want to calculate a single statistic that appropriately weights the FNR/FPR
        weights = np.array([FNR_FPR_weight(s, FNR_[j], FPR_[j]) for j in range(len(FNR_))])
        weighted_rates.append(weights)

    # Hopefully these arrays aren't jagged
    weighted_rates = np.array(weighted_rates)
    avg_cov = np.array(avg_cov)

    # Sum the weighted rates across sparsity, divide by number of sparsities
    try:
        weighted_rates = np.mean(weighted_rates, axis = 0)
        avg_cov = avg_cov[0, :]
    except:
        # lists are uneven in size --> need to take the intersection (with duplicates) of all elements
        avg_cov, trimmed_idxs = array_intersection(avg_cov)
        pdb.set_trace()
        avg_cov = avg_cov[0, :]
        weighted_rates = np.array([weighted_rates[j][trimmed_idxs[j]] for j in range(weighted_rates.size)])
        weighted_rates = np.mean(weighted_rates, axis = 0)

    # Sort by avg_cov
    cov_order = np.argsort(avg_cov)
    weighted_rates = weighted_rates[cov_order]
    avg_cov = avg_cov[cov_order]

    # Do not do moving averaging first
#    return moving_average(weighted_rates), moving_average(avg_cov)
    return weighted_rates, avg_cov

def FNR_FPR_weight(sparsity, FNR, FPR):

    # Given the coordinates in the (FNR, FPR) plane, calculate the line integral from the origin to the point in question

    weight_fn = lambda u, x, y, s : ((1 - s) * y * u + s * x * u)/(s * (2 - x * u - y * u) + y * u)
    weight, _  = integrate.quad(weight_fn, 0, 1, args = (FNR, FPR, sparsity))
    return weight

def moving_average(a, n=3):
    ret = np.nancumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# More simple plotting of the selection accuracy vs the average correlation
def sa_v_corr(df, n_features):
    sparsity = np.unique(df['sparsity'])

    sa = []
    avg_cov = []
    for i, s, in enumerate(sparsity):
        df_ = apply_df_filters(df, sparsity = s)

        # Calculate average covariance
        cov_params, rep_idxs = unique_cov_params(df_)
        avg_cov.append(np.array([calc_avg_cov(n_features, **cp) for cp in cov_params]))
        sa.append(average_fields(df_, ['sa'], rep_idxs)[0])

    # Hopefully these arrays aren't jagged
    sa = np.array(sa)
    avg_cov = np.array(avg_cov)
    # Sum the weighted rates across sparsity, divide by number of sparsities
    try:
        sa = np.mean(sa, axis = 0)
        avg_cov = avg_cov[0, :]
    except:
        # lists are uneven in size --> need to take the intersection (with duplicates) of all elements
        avg_cov, trimmed_idxs = array_intersection(avg_cov)
        avg_cov = avg_cov[0, :]
        sa = np.array([sa[j][trimmed_idxs[j]] for j in range(sa.size)])
        sa = np.mean(sa, axis = 0)

    # Sort by avg_cov
    cov_order = np.argsort(avg_cov)
    sa = sa[cov_order]
    avg_cov = avg_cov[cov_order]
    # Do not do moving averaging first
#    return moving_average(weighted_rates), moving_average(avg_cov)
    return sa, avg_cov
