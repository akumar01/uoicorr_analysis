import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from postprocess_utils import group_dictionaries, average_fields
import pdb


# Calculate the average covariance given the 5 parameters that define
# an interpolated covariance matrix
# Trim off the diagonal component
def calc_avg_cov(p, correlation, block_size, L, t):

    # Average correlation of th
    c1 = correlation * (block_size - 1)/p
    c2 = 1/p**2 * (2 * (np.exp(1/L) * (p + np.exp(-p/L) -1 ) - p)\
                                                    /(np.exp(1/L) - 1)**2)

    return (1- t) * c1 + t * c2

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
def FNR_FPR_scatter(axis, df, color):

    cov_params, rep_idxs = unique_cov_params(df)
    avg_cov = [calc_avg_cov(500, **cp) for cp in cov_params]
    FNR, FPR = average_fields(df, ['FNR', 'FPR'], rep_idxs)
    c = [colors.to_rgba(color, alpha = np.power(avgcov, 0.35)) for avgcov in avg_cov]
    s = axis.scatter(FNR, FPR, c = c)
    return axis, s


def marginalize(df, dep, *indep):

    # Return a list of values of dep, averaged over all fields not
    # specified in *indep (list of strings)

    # Need to get unique values of combinations of *indep
    values_of_interest = df[list(indep)].to_dict('records')
    unique_values, rep_idxs = group_dictionaries(values_of_interest, None)
    averaged_values = average_fields(df, [dep], rep_idxs)

    return unique_values, averaged_values

# Plot lineplots of y marginialized over sparsity as a function of x
def sparsity_marginalized_1D(axis, df, color, x, y):

    # Average over all fields not specified by x and y
    xvals, yvals = marginalize(df, y, x)
    p = axis.plot(xvals, yvals, '-o', c = color)
    return p

# Plot things
def sparsity_marginalized_2D(axis, df, x, y, z): 

    xvals, yvals, zvals = marginalize(df, z, x, y)
    h = axis.pcolormesh(y, x, z)
    return h

def error_probability_1D(axis, df, x, error_threshold):

    # Report the fraction of elements in repeated elements of unique values of x that
    # pass the error threshold

    values_of_interest = df[list(indep)].to_dict('records')
    unique_values, rep_idxs = group_dictionaries(values_of_interest, None)
    error_probabilities = threshold(df, rep_idxs)
    axis.plot(unique_values, error_probabilities)

def error_probability_2D(axis, df, x, y, error_threshold):

    values_of_interest = df[list(indep)].to_dict('records')
    unique_values, rep_idxs = group_dictionaries(values_of_interest, None)
    error_probabilities = threshold(df, rep_idxs)
    axis.plot(unique_values, error_probabilities)

# Likely best suited by some kind of boxplot here
def bias_variance_plots()



fig, ax = plt.subplots(3, 1, figsize = (15, 15))

################# SNR 100 ################################################################

en_ = apply_df_filters(en, kappa=100, selection_method='BIC')
lasso_ = apply_df_filters(lasso, kappa=100, selection_method='BIC', betadict = {'betawidth' : np.inf})
uoil_ = apply_df_filters(uoil,  kappa=100, selection_method='BIC', betadict = {'betawidth' : np.inf})
mcp_ = apply_df_filters(mcp, kappa=100, selection_method='BIC', betadict = {'betawidth' : np.inf})
scad_ = apply_df_filters(scad, kappa=100, selection_method='BIC', betadict = {'betawidth' : np.inf})

# Extract weights
enw, enc = weighted_rates(en_, 500)
lassow, lassoc = weighted_rates(lasso_, 500)
uoilw, uoilc = weighted_rates(uoil_, 500)
mcpw, mcpc = weighted_rates(mcp_, 500)
scadw, scadc = weighted_rates(scad_, 500)

ax[0].plot(lassoc, lassow, '-o', c = c4[0])
ax[0].plot(enc, enw, '-o', c = c5[0])
ax[0].plot(scadc, scadw, '-o', c = c1[0])
ax[0].plot(mcpc, mcpw, '-o', c = c2[0])
ax[0].plot(uoilc, uoilw, '-o', c = c3[0])
ax[0].set_title('SNR 100', fontsize = 16)

# SNR 5 #
en_ = apply_df_filters(en, kappa=5, selection_method='BIC', betadict = {'betawidth' : np.inf})
lasso_ = apply_df_filters(lasso, kappa=5, selection_method='BIC', betadict = {'betawidth' : np.inf})
uoil_ = apply_df_filters(uoil,  kappa=5, selection_method='BIC', betadict = {'betawidth' : np.inf})
mcp_ = apply_df_filters(mcp, kappa=5, selection_method='BIC', betadict = {'betawidth' : np.inf})
scad_ = apply_df_filters(scad, kappa=5, selection_method='BIC', betadict = {'betawidth' : np.inf})

# Extract weights
enw, enc = weighted_rates(en_, 500)
lassow, lassoc = weighted_rates(lasso_, 500)
uoilw, uoilc = weighted_rates(uoil_, 500)
mcpw, mcpc = weighted_rates(mcp_, 500)
scadw, scadc = weighted_rates(scad_, 500)

ax[1].plot(lassoc, lassow, '-o', c = c4[0])
ax[1].plot(enc, enw, '-o', c = c5[0])
ax[1].plot(scadc, scadw, '-o', c = c1[0])
ax[1].plot(mcpc, mcpw, '-o', c = c2[0])
ax[1].plot(uoilc, uoilw, '-o', c = c3[0])
ax[1].set_title('SNR 5', fontsize = 16)

# SNR 2
# Filter to a single betawidth, kappa
en_ = apply_df_filters(en, kappa=2, selection_method='BIC', betadict = {'betawidth' : np.inf})
lasso_ = apply_df_filters(lasso, kappa=2, selection_method='BIC', betadict = {'betawidth' : np.inf})
uoil_ = apply_df_filters(uoil,  kappa=2, selection_method='BIC', betadict = {'betawidth' : np.inf})
mcp_ = apply_df_filters(mcp, kappa=2, selection_method='BIC', betadict = {'betawidth' : np.inf})
scad_ = apply_df_filters(scad, kappa=2, selection_method='BIC', betadict = {'betawidth' : np.inf})

# Extract weights
enw, enc = weighted_rates(en_, 500)
lassow, lassoc = weighted_rates(lasso_, 500)
uoilw, uoilc = weighted_rates(uoil_, 500)
mcpw, mcpc = weighted_rates(mcp_, 500)
scadw, scadc = weighted_rates(scad_, 500)

ax[2].plot(lassoc, lassow, '-o', c = c4[0])
ax[2].plot(enc, enw, '-o', c = c5[0])
ax[2].plot(scadc, scadw, '-o', c = c1[0])
ax[2].plot(mcpc, mcpw, '-o', c = c2[0])
ax[2].plot(uoilc, uoilw, '-o', c = c3[0])
ax[2].set_title('SNR 2', fontsize = 16)
ax[2].set_xlabel('Average Correlation', fontsize = 14)

ax[2].legend(['Lasso', 'Elastic Net', 'SCAD', 'MCP', r'$UoI_{Lasso}$'], loc = 'lower right')
plt.savefig('pathint_snr.pdf', bbox_inches = 'tight', pad_inches = 0)
