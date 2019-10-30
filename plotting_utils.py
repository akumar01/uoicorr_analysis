import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from postprocess_utils import group_dictionaries, average_fields, apply_df_filters, moving_average
from utils import gen_covariance
from scipy import interpolate
import pdb
import collections
import itertools
import time

# Calculate the average covariance given the 5 parameters that define
# an interpolated covariance matrix
# Trim off the diagonal component
def calc_avg_cov(p, correlation, block_size, L, t):

    # Average correlation of th
    c1 = correlation * (block_size - 1)/p
    c2 = 1/p**2 * (2 * (np.exp(1/L) * (p + np.exp(-p/L) -1 ) - p)\
                                                    /(np.exp(1/L) - 1)**2)

    return (1- t) * c1 + t * c2

def calc_quantile(df, quantile, fields, rep_idxs):

    results = []
    for i, field in enumerate(fields):
        values = np.zeros(len(rep_idxs))
        for j, rep_idx in enumerate(rep_idxs):
            values[j] = np.quantile(df.iloc[rep_idx][field], quantile)
        results.append(values)
    return results

def n_scaling(axis, df):
    pass
    # P


# Returns unique instances of cov params under the new format where these are represented as
# separate fields
def unique_cov_params(df):

    # transform back to a list of dictionaries:
    cov_params = df[['correlation', 'block_size', 'L', 't']].to_dict('records')
    # use the existing functionality
    unique_cov_params, cov_idxs = group_dictionaries(cov_params, None)
    unique_cov_dicts = []
    for ucp in unique_cov_params:
        ucd = {'correlation' : ucp[0], 'block_size' : ucp[1], 'L' : ucp[2], 't': ucp[3]}
        unique_cov_dicts.append(ucd)

    return unique_cov_dicts, cov_idxs

# Send in a filtered dframe and add a scatter plot to the axis
# axis to add the scatter plot to
# dataframe that contains data to be plotted
# color: color for the scatter plot (will set opacity via average correlation)
def FNR_FPR_scatter(axis, df, color, marker):

    cov_params, rep_idxs = unique_cov_params(df)
    avg_cov = [calc_avg_cov(500, **cp) for cp in cov_params]
    FNR, FPR = average_fields(df, ['FNR', 'FPR'], rep_idxs)
    c = [colors.to_rgba(color, alpha = np.power(avgcov, 0.35)) for avgcov in avg_cov]
    s = axis.scatter(FNR, FPR, c = c, marker = marker)
    return axis, s


def marginalize(df, dep, *indep):

    # Return a list of values of dep, averaged over all fields not
    # specified in *indep (list of strings)

    # Need to get unique values of combinations of *indep
    values_of_interest = df[list(*indep)].to_dict('records')
    unique_values, rep_idxs = group_dictionaries(values_of_interest, None)
    averaged_values, std_values = average_fields(df, [dep], rep_idxs, True)
    return unique_values, averaged_values, std_values

# Instead of taking the average, return the quantile value
def marginalize_q(df, quantile, dep, *indep):

    values_of_interest = df[list(*indep)].to_dict('records')
    unique_values, rep_idxs = group_dictionaries(values_of_interest, None)
    quantile_values = calc_quantile(df, quantile, [dep], rep_idxs)
    return unique_values, quantile_values

# Fix a threshold. How many of the reps exceed that threshold?
def error_probability():
    pass



# Plot lineplots of y marginialized over sparsity as a function of x
def marginalized_1D(axis, df, color, x, y):

    # Average over all fields not specified by x and y
    xvals, yvals, yerr = marginalize(df, y, x)

    # xvals need to be unpacked and sorted
    xvals = np.array([x for sublist in xvals for x in sublist])
    xorder = np.argsort(xvals)
    xvals = xvals[xorder]

    yvals = yvals[0]
    yvals = yvals[xorder]

    yerr = yerr[0]
    yerr = yerr[xorder]

    p = axis.errorbar(xvals, yvals, yerr=yerr, c = color)
    return p

# Plot things
# def sparsity_marginalized_2D(axis, df, x, y, z):

#     xvals, yvals, zvals = marginalize(df, z, x, y)
#     h = axis.pcolormesh(y, x, z)
#     return h
def sparsity_corr_2D(axis, df, z, use_eig_bound=False):

    # Will plot the quantity z against either the average
    # correlation or eig_bound vs. the sparsity

    # To best use marginalize, we first filter by sparsity
    # and then compute things per sparsity
    sparsity = np.unique(df['sparsity'].values)[3:]
    p = df.iloc[0]['n_features']

    # Truncate the first 3 sparsities since they do not have a full complement of
    # correlated designs

    zarray = []
    xarray = []

    for i, s in enumerate(sparsity):
        k = int(s * p)
        df_ = apply_df_filters(df, sparsity=s)
        x, zvals, _ = marginalize(df_, z, ['correlation', 'block_size', 'L', 't'])
#        x, zvals = error_probability()
        if use_eig_bound:
            corr = np.array([calc_eigen_bound(p, k, *xx) for xx in x])
        else:
            corr = np.array([calc_avg_cov(p, *xx) for xx in x])

        xarray.append(corr)
        zvals = zvals[0]
        zarray.append(zvals)

    xarray = np.array(xarray)
    zarray = np.array(zarray)

    # We are missing some values, mask them
    if xarray.ndim != 2:
        xarray, zarray = mask_jagged_array(xarray, zarray)

    # Sort each column
    ordering = np.argsort(xarray, axis = 1)
    xarray = np.take_along_axis(xarray, ordering, 1)
    zarray = np.take_along_axis(zarray, ordering, 1)


    # As a dirty 1st pass, just make colorplots
    # axis.pcolormesh(xarray, sparsity, zarray, vmin = 0, vmax = 1, cmap = 'Greys_r',
    #                 shading='gouraud')

    # Plot 3D surfaces of what results
    # yarray = np.tile(sparsity[:, np.newaxis], 80)
    # axis.plot_surface(xarray, yarray, zarray)

# eig bound df: A dataframe giving the eigenvalue bounds for combination
# of cov params

def sparsity_corr_2D_2(axis, df, z, cut_outliers=True, eig_bound_df=None):

    # Take the maximum across the dataframes supplied

    # Truncate the first 3 sparsities since they do not have a full complement of
    # correlated designs

    sparsity = np.unique(df['sparsity'].values)[3:]
    p = df.iloc[0]['n_features']

    zarray = []
    xarray = []

    for i, s in enumerate(sparsity):
        k = int(s * p)
        df_ = apply_df_filters(df, sparsity=s)
        x, zvals, _ = marginalize(df_, z, ['correlation', 'block_size', 'L', 't'])
#        x, zvals = error_probability()
        corr = np.array([calc_avg_cov(p, *xx) for xx in x])
        # rho = np.array([apply_df_filters(eig_bound_df, correlation = xx[0], block_size = xx[1],
        #                                  L = xx[2], t = xx[3], k = k).iloc[0]['rho']
        #                 for xx in x])

        xarray.append(corr)
#        xarray.append(rho)
        zvals = zvals[0]
        zarray.append(zvals)

    xarray = np.array(xarray)
    zarray = np.array(zarray)

    # We are missing some values, mask them
    if xarray.ndim != 2:
        xarray, zarray = mask_jagged_array(xarray, zarray)

    # Sort each column
    ordering = np.argsort(xarray, axis = 1)
    xarray = np.take_along_axis(xarray, ordering, 1)
    zarray = np.take_along_axis(zarray, ordering, 1)

    if cut_outliers:
        xarray = np.delete(xarray, [24, 41, 42, 43, 57, 58, 59, 60], axis = 1)
        zarray = np.delete(zarray, [24, 41, 42, 43, 57, 58, 59, 60], axis = 1)



    zinterp = interpolate.interp2d(xarray[0, :], sparsity, zarray)
    # Denser sampling along sparsity axis
#    sparsity_dense = np.linspace(np.min(sparsity), 1, 40)
 #   zdense = zinterp(xarray[0, :], sparsity_dense)

    # Apply moving average smoothing to the correlation direction
    zdense = np.array([moving_average(zarray[i, :]) for i in range(zarray.shape[0])])
    # Chop off first and last element to match new shape
    xdense = xarray[0, 1:-1]
    h = axis.pcolormesh(xdense, sparsity, zdense, shading='gouraud', vmin = 0, vmax = 1, cmap='bone')

    return h

def mask_jagged_array(ref_array, other_array):

    # Use a subarray of ref_array with the largest size a
    # reference. If other subarrays are missing values or
    # do not have the same number of that value, then put it in
    sizes = [subarray.size for subarray in ref_array]
    ref_set = ref_array[np.argmax(sizes)]

    # Need this guy to be mutable
    ref_array = list(ref_array)
    other_array = list(other_array)

    for i in range(len(ref_array)):

        ref_array[i] = list(ref_array[i])
        other_array[i] = list(other_array[i])

        if len(ref_array[i]) < len(ref_set):
            cnt1 = collections.Counter(ref_set)
            cnt2 = collections.Counter(ref_array[i])

            # What entries in cnt1 are not present in cnt2?
            diff_dict = {k : v for k, v in cnt1.items() if k not in cnt2.keys()}
            # What entries in cnt1 occur a different number of times than in cnt2?
            diff_dict.update({k : v - cnt2[k] for k, v in cnt1.items() if k in cnt2.keys()
                                            and v > cnt2[k]})

            [ref_array[i].extend(itertools.repeat(k, v)) for k, v in diff_dict.items()]
            [other_array[i].extend(itertools.repeat(np.nan, v)) for v in diff_dict.values()]
    # Should return square 2D arrays
    return np.array(ref_array), np.array(other_array)


# Likely best suited by some kind of boxplot here
def bias_variance_plots():
    pass
