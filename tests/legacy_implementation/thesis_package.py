"""Unchanged utils package from Thesis used for consistency checks."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:14:03 2020

@author: max
"""

import math as m

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.special import erfinv, loggamma
from scipy.stats import argus, beta, iqr, moment, rdist
from sklearn.cluster import (
    DBSCAN,
    AgglomerativeClustering,
    KMeans,
    MeanShift,
    estimate_bandwidth,
)
from sklearn.mixture import GaussianMixture as GMM
from sklearn.neighbors import KernelDensity as KD


# Print iterations progress
def printProgressBar(
    iteration,
    total,
    prefix="Progress: ",
    suffix="",
    decimals=1,
    length=20,
    fill="█",
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    try:
        if total == 0:
            percent = 0
            filledLength = length
        else:
            percent = ("{0:." + str(decimals) + "f}").format(
                100 * (iteration / float(total))
            )
            filledLength = int(length * iteration // total)
        bar = fill * filledLength + "-" * (length - filledLength)
        print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
        # Print New Line on Complete
        if iteration == total:
            print()
    except:
        return


def importEurostoxxDow(test=False):
    # Import the Eurostoxx Dow Jones Futures Dataset with correct specifications.
    if test:
        data = pd.read_excel(
            "../Data Eurostoxx Dow Futures/Dow_Eurostoxx_Futures.xlsx",
            usecols="B:D",
            nrows=100,
            header=0,
            index_col=0,
            skiprows=[0, 2],
        )
    else:
        data = pd.read_excel(
            "../Data Eurostoxx Dow Futures/Dow_Eurostoxx_Futures.xlsx",
            usecols="B:D",
            header=0,
            index_col=0,
            skiprows=[0, 2],
        )
    data.drop(data.tail(1).index, inplace=True)
    return data


def importEurostoxxDow_rankOrder():
    return pd.DataFrame(
        rank_remap_gaussian(importEurostoxxDow()), columns=["Eurostoxx", "Dow Jones"]
    )


def import_May15_Data(test=False):
    path = "../Data/Daten_May15.xlsx"
    if test:
        data = pd.read_excel(
            path,
            usecols="B:U",
            header=0,
            index_col=0,
            skiprows=[0, 1, 2, 3, 4, 5, 6],
            nrows=100,
            converters={"B": pd.to_datetime},
        )
    else:
        data = pd.read_excel(
            path, usecols="B:U", header=0, index_col=0, skiprows=[0, 1, 2, 3, 4, 5, 6]
        )
    data.columns = [
        "ES1_Close",
        "ES1_Volume",
        "ES1_Number",
        "ES1_Ticks",
        "VG1_Close",
        "VG1_Volume",
        "VG1_Number",
        "VG1_Ticks",
        "H1_Close",
        "H1_Volume",
        "H1_Number",
        "H1_Ticks",
        "NK1_Close",
        "NK1_Volume",
        "NK1_Number",
        "NK1_Ticks",
        "CO1_Close",
        "CO1_Volume",
        "CO1_Number",
    ]
    return data


"""
def entropy(data, bins):
    #Compute information entropy of time series
    if type(data) == pd.core.frame.DataFrame:
        data = data.to_numpy()
    if type(data) == list:
        data = np.array(data)

    length = float(data.shape[0]) # Number of data points per variable

    try:
        dim = data.shape[1] # Number of variables
    except IndexError:
        dim = 1
    if dim == 1:
        p = np.histogram(data, bins)[0]

        #Normalization
        p = p / length

        # Reassign  bins with zero entries to value 1 for finite results
        p = np.where(p == 0, 1, p)

        # Compute entropy
        ent = - np.sum(p* np.log(p))
    else:
        ent = []
        for i in range(dim):
            if bins.ndim > 1 or bins.dtype == np.dtype('O'):
                input_bins = bins[i].flatten()
            else:
                input_bins = bins
            ent_ = entropy(data[:, i], input_bins)
            ent.append(ent_)
    return ent"""


def entropy(data, bins, new_est=False):
    if data.ndim == 1:
        N = len(data)
        hist = np.histogram(data, bins)[0]
        if new_est:
            hist = OneD_hist_to_new_estimator(hist, bins)
        else:
            hist = hist / N
        hist = np.where(hist == 0, 1, hist)
        hist *= np.log(hist)
        return -np.sum(hist)
    elif data.ndim == 2:
        ents = np.zeros(data.shape[1])
        for i in range(len(ents)):
            if type(bins) == list:
                input_bins = bins[i]
            else:
                input_bins = bins
            ents[i] = entropy(data[:, i], input_bins, new_est=new_est)
        return ents
    else:
        raise ValueError("Data must be of dimension [timesteps x Variables].")


def joint_entropy(data, bins, new_est=False):
    # Compute joint information entropy for each pair
    # bins can be int for nbins in x & y , [int, int] for nxbins & nybins,
    # array for edges in x& y ,[array, array] for  xedges& yedges
    # Returns array with columns row given colum: [[1_1,..., 1_N], ... [N_1,...,N_N]] joint entropies
    if type(data) == pd.core.frame.DataFrame:
        data = data.to_numpy()
    try:
        dim = data.shape[1]  # Number of variables
        length = float(data.shape[0])  # Number of data points
    except IndexError:
        print(
            "Joint Entropy Error: Cannot compute joint entropy with only one variable given."
        )
        return 0
    jent = np.zeros((dim, dim), dtype=float)
    for i in range(dim):
        for j in range(i + 1):
            if type(bins) == list or bins.ndim > 1 or bins.dtype == np.dtype("O"):
                input_bins = [bins[i].flatten(), bins[j].flatten()]
            else:
                input_bins = bins
            # Calculate joint probability distributions
            p_xy = np.histogram2d(data[:, i], data[:, j], input_bins)[0]
            # Normalize by time series length
            if new_est:
                if len(input_bins) == 2:
                    p_xy = TwoD_hist_to_new_estimator(p_xy, input_bins)
                else:
                    p_xy = TwoD_hist_to_new_estimator(p_xy, [input_bins, input_bins])
            else:
                p_xy /= length

            # Ensure zero bins to not be considered in entropy estimate
            p_xy = np.where(p_xy == 0, 1, p_xy)

            # Calculate joint information entropy
            jent[i, j] = -np.sum(p_xy * np.log(p_xy))
            jent[j, i] = jent[i, j]
    return jent


def conditional_entropy(data, bins):
    # Calculate conditional entropy of time series via chain rule:
    # H(Y|X) = H(X,Y) - H(X)
    # Ouptut of form [[ 0|0, 0|1 ], [1|0 , 1|1]]

    if type(data) == pd.core.frame.DataFrame:
        data = data.to_numpy()
    try:
        dim = data.shape[1]  # Number of data points
    except IndexError:
        print(
            "Conditional Entropy Error: Cannont compute  with only one variable given."
        )
        return 0
    cent = np.zeros((dim, dim))
    h_xy = joint_entropy(data, bins)
    h_x = entropy(data, bins)
    for i in range(dim):
        for j in range(i + 1):
            cent[i, j] = h_xy[i, j] - h_x[i]
            cent[j, i] = h_xy[j, i] - h_x[j]
    return cent


def mutual_information(data, bins, norm=1):
    """
    # Calculate mutual information of time series
    # I(X,Y) = H(X) + H(Y) - H(X,Y)
    # data = [timesteps x dimension]
    # bins = number of bins for histogram approximation of PDF
    # norm = [1 if result should be normalized between 0 and 1, 0 else]
    # normalized form: I(X,Y) = (H(X) + H(Y) - H(X,Y))/sqrt(H(X)*H(Y))
    """
    if type(data) == pd.core.frame.DataFrame:
        data = data.to_numpy()
    try:
        dim = data.shape[1]  # Number of data points
    except IndexError:
        print(
            "Mutual Information Error: Cannont compute  with only one variable given."
        )
        return 0
    h_xy = joint_entropy(data, bins)
    h_x = entropy(data, bins)

    mi = np.zeros((dim, dim))

    for i in range(dim):
        for j in range(i + 1):
            if norm:
                mi[i, j] = np.divide(
                    h_x[i] + h_x[j] - h_xy[i, j], np.sqrt(np.multiply(h_x[i], h_x[j]))
                )
                mi[j, i] = mi[i, j]
            else:
                mi[i, j] = h_x[i] + h_x[j] - h_xy[i, j]
                mi[j, i] = mi[i, j]
    return mi


def cross_correlation(data, tau=0):
    """
    Calculate cross correlation of time series.
    # data: Input. Can be pandas dataframe or array.
    # tau: Shift of data
    # Returns array of cross correlations
    """
    if type(data) == pd.core.frame.DataFrame:
        data = data.to_numpy()
    try:
        dim = data.shape[1]  # Number of data points
    except IndexError:
        print("Cross correlation Error: Cannont compute  with only one variable given.")
        return 0
    cross = np.zeros((dim, dim))
    for i in range(dim):
        data1 = data[:, i]
        if tau > 0:
            data1 = np.roll(data1, tau)  # shift data if tau specified
        iMean = np.mean(data1)
        s_i = 0.0
        for t in range(len(data1)):
            s_i += (data1[t] - iMean) ** 2
        for j in range(i + 1):
            cov = 0.0
            data2 = data[:, j]
            jMean = np.mean(data2)
            s_j = 0.0
            for t in range(max(len(data1), len(data2))):
                sqrt_s_j = data2[t] - jMean
                s_j += sqrt_s_j**2
                cov += (data1[t] - iMean) * sqrt_s_j

            cross[i, j] = cov / (np.sqrt(s_i * s_j))
            cross[j, i] = cross[i, j]

    return cross


def OneD_hist_to_new_estimator(hist, bins):
    vols = [bins[i + 1] - bins[i] for i in range(0, len(bins) - 1)]
    span = np.sum([vols[i] if hist[i] > 0 else 0 for i in range(len(hist))])
    ps = 1.0 * np.zeros_like(hist)
    N = np.sum(hist)

    for i in range(len(hist)):
        ps[i] = hist[i] / N * vols[i] / span
    return ps / np.sum(ps)


def TwoD_hist_to_new_estimator(hist, bin_list):
    bin_widths = [
        [bins[i + 1] - bins[i] for i in range(len(bins) - 1)] for bins in bin_list
    ]
    vols = 1.0 * np.ones_like(hist)
    N = np.sum(hist)
    for i in range(vols.shape[0]):
        for j in range(vols.shape[1]):
            vols[i, j] = bin_widths[0][i] * bin_widths[1][j]
    total_occupied_volume = np.sum(vols[hist > 0])
    vs = vols.flatten()
    flat_hist = hist.flatten()
    ps = 1.0 * np.zeros_like(flat_hist)

    for i in range(len(flat_hist)):
        p = flat_hist[i] / N * (vs[i] / total_occupied_volume) ** (1 / 2)
        ps[i] = p
    return np.reshape(ps, hist.shape) / np.sum(ps)


def ThreeD_hist_to_new_estimator(hist, bin_list):
    bin_widths = [
        [bins[i + 1] - bins[i] for i in range(len(bins) - 1)] for bins in bin_list
    ]
    vols = 1.0 * np.zeros_like(hist)
    N = np.sum(hist)
    for i in range(vols.shape[0]):
        for j in range(vols.shape[1]):
            for k in range(vols.shape[2]):
                vols[i, j, k] = bin_widths[0][i] * bin_widths[1][j] * bin_widths[2][k]
    total_occupied_volume = np.sum(vols[hist > 0])
    vs = vols.flatten()
    flat_hist = hist.flatten()
    ps = 1.0 * np.zeros_like(flat_hist)

    for i in range(len(flat_hist)):
        p = flat_hist[i] / N * (vs[i] / total_occupied_volume) ** (1 / 3)
        ps[i] = p
    return np.reshape(ps, hist.shape) / np.sum(ps)


def normalised_transfer_entropy(data, bins, lag, new_est=False):
    """
    Calculate normalised transfer entropy as
    1 - H(Y_t | Y_t_lag, X_t_lag)/ H(Y_t | Y_t_lag)
    with H(Y_t | Y_t_lag, X_t_lag) = H(Y_t, Y_t_lag, X_t_lag) - H(Y_t_lag, X_t_lag)
    and H(Y_t | Y_t_lag) = H(Y_t, Y_t_lag) - H(Y_t_lag)
    :param data:
    :param bins:
    :param lag:
    :return:
    """
    # NOT UPDATED TO TAKE INDIVIDUAL BINS FOR VARIABLES.
    dim = data.shape[1]
    dat = data[lag:, :]
    datLag = data[:-lag, :]
    N = dat.shape[0]

    tent = np.zeros((dim, dim))

    # H(Y_t_lag, X_t_lag)
    h_ylag_xlag = joint_entropy(datLag, bins, new_est=new_est)

    # H(Y_t_lag)
    h_ylag = entropy(datLag, bins, new_est=new_est)

    for i in range(dim):
        # H(Y_t, Y_t_lag)
        p_y_ylag = np.histogramdd(
            np.column_stack([dat[:, i], datLag[:, i]]), [bins, bins]
        )[0]
        if new_est:
            p_y_ylag = TwoD_hist_to_new_estimator(p_y_ylag, [bins, bins])
        else:
            p_y_ylag = p_y_ylag / N
        p_y_ylag = np.where(p_y_ylag == 0, 1, p_y_ylag)
        prods1 = p_y_ylag * np.log(p_y_ylag)
        h_y_ylag = -np.sum(prods1)

        for j in range(dim):
            # H(Y_t, Y_t_lag, X_t_lag)
            p_xyz = np.histogramdd(
                np.column_stack([dat[:, i], datLag[:, i], datLag[:, j]]),
                [bins, bins, bins],
            )[0]
            if new_est:
                p_xyz = ThreeD_hist_to_new_estimator(p_xyz, [bins, bins, bins])
            else:
                p_xyz = p_xyz / N
            p_xyz = np.where(p_xyz == 0, 1, p_xyz)
            prods2 = p_xyz * np.log(p_xyz)
            h_y_ylag_xlag = -np.sum(prods2)

            numerator = h_y_ylag_xlag - h_ylag_xlag[i, j]
            denumerator = h_y_ylag - h_ylag[i]
            if denumerator == 0:
                tent[i, j] = 0
            else:
                tent[i, j] = round(1 - numerator / denumerator, 10)
    return tent


def logN_normalised_transfer_entropy(data_0, bins, lag, new_est=False):
    if type(bins) == list:
        te = transfer_entropy(data_0, bins, lag, new_est=new_est)
        for i in range(te.shape[0]):
            te[i] = te[i] / np.log(len(bins[i]) - 1)
        return te
    else:
        return transfer_entropy(data_0, bins, lag, new_est=new_est) / np.log(
            len(bins) - 1
        )


def transfer_entropy(data_0, bins, lag, new_est=False):
    """
    # Calculate transfer entropy of time series:
    # T_x->y = H(Y_t|Y_t-1) - H(Y_t-1|Y_t_1;X_t-1)
    # Express as conditional mutual information I(Y_t;X_t-1|Y_t-1)
    # I(Y_t;X_t-1|Y_t-1) = H(Y_t,Y_t-1) + H(X_t-1,Y_t-1) - H(Y_t,Y_t-1,X_t-1) - H(Y_t-1)
    # Inputs:
    # data = [timesteps x dimension]
    # bins = number of bins for histogram approximation of PDF
    # lag = shift of time series

    Returns [dim x dim] matrix with dim the number of variables in data_0
    """
    if type(data_0) == pd.core.frame.DataFrame:
        data_0 = data_0.to_numpy()
    try:
        dim = data_0.shape[1]  # Number of data points
    except IndexError:
        print("Transfer Entropy Error: Cannot compute with only one variable given.")
        return 0
    if type(bins) != list:
        bins = [bins for b in range(dim)]
    # Set up data and lagged data
    data = data_0[lag:, :]
    dataLag = data_0[:-lag, :]
    length = data.shape[0]
    # Calc H(X_t-1,Y_t-1) and H(X_t-1)
    h_xy_lag = joint_entropy(dataLag, bins, new_est=new_est)
    h_x_lag = entropy(dataLag, bins, new_est=new_est)

    tent = np.zeros((dim, dim))

    for i in range(dim):
        # Calc H(Y_t,Y_t-1), for comment see bivariate joint entropy function
        p_xy = np.histogram2d(data[:, i], dataLag[:, i], bins[i])[0]
        if new_est:
            p_xy = TwoD_hist_to_new_estimator(p_xy, [bins[i], bins[i]])
        else:
            p_xy = np.divide(p_xy, length)
        p_xy = np.where(p_xy == 0, 1, p_xy)
        prods1 = np.multiply(p_xy, np.log(p_xy))
        h_y_ylag = -np.sum(prods1)

        for j in range(dim):
            # Calculate H(Y_t,Y_t-1,X_t-1)
            p_xyz = np.histogramdd(
                np.column_stack([data[:, i], dataLag[:, i], dataLag[:, j]]),
                [bins[i], bins[i], bins[j]],
            )[0]
            if new_est:
                p_xyz = ThreeD_hist_to_new_estimator(p_xyz, [bins[i], bins[i], bins[j]])
            else:
                p_xyz = np.divide(p_xyz, length)
            p_xyz = np.where(p_xyz == 0, 1, p_xyz)
            prods2 = np.multiply(p_xyz, np.log(p_xyz))
            h_y_ylag_xlag = -np.sum(prods2)

            # Construct transfer entropy
            tent[i, j] = h_y_ylag + h_xy_lag[i, j] - h_y_ylag_xlag - h_x_lag[i]
    return tent


def slidingTentData(timeSeriesDf, window_size, bins, lag=1, test=False, stepsize=1):
    """
    Returns a data frame of sliding time window transfer entropy calculations.

    #timeSeriesDf: Dataframe from wich pairwise transfer Entropy for all columns is computed
    #window_size: Size of previous values contributing to the transfer entropy.
    #bins: Bins for transfer entropy calculation
    #lag: Lag for transfer entropy calculation
    #test: if true, alter code such that much less iterations are undertaken and code runs faster
    #stepsize: difference between datapoints for which TE in window is calculated
    """
    print("Sliding Tent Data Generation")

    if test:
        stepsize = window_size

    # Create transfer entropy data frame
    tscols = timeSeriesDf.columns
    slidingTentData_cols = []
    for i in range(len(tscols)):
        for j in range(len(tscols)):
            if i == j:
                next
            else:
                slidingTentData_cols.append(tscols[i] + " -> " + tscols[j])
    slidingTentData = pd.DataFrame(columns=slidingTentData_cols)

    # Create Array to store indices of first element in sliding window for each iteration
    slidingTentIndex = []
    for i in np.arange(window_size, len(timeSeriesDf), stepsize):
        printProgressBar(i, len(timeSeriesDf) - 1)
        slidingTentIndex.append(timeSeriesDf.iloc[i - window_size : i, :].index[0])
        inter_df = transfer_entropy(
            timeSeriesDf.iloc[i - window_size : i, :], bins, lag
        )

        nextRowDict = {}
        col_index = 0
        for i in range(len(tscols)):
            for j in range(len(tscols)):
                if i == j:
                    next
                else:
                    nextRowDict[slidingTentData_cols[col_index]] = inter_df[i, j]
                    col_index += 1

        slidingTentData = slidingTentData.append(nextRowDict, ignore_index=True).copy()
    slidingTentData.index = slidingTentIndex
    return slidingTentData


def ft_surrogate(inptData, rand_phases=False, seed=False):
    """
    # Calculate Fourier Transform (FT) Surrogates of the original time
    # series. This keeps the linear properties of the timeseries like def ft_surrogate(data, rand_phases):
    # Input: data = [timesteps x dimension]
    # Random Phases need to have same dimensions as data.
    # seed: Add seed to force repeatability. Has no effect if random phases are provided
    """
    data = inptData.copy()
    try:
        idx = data.index
    except AttributeError:
        idx = pd.RangeIndex(start=0, stop=len(data), step=1)

    # Seed data if seed argument is given.
    if seed:
        np.random.seed(seed)
        if rand_phases:
            print(
                "Warning: Both a seed value and random phases were provided. The seed might have no effect."
            )

    try:
        cols = data.columns
    except AttributeError:
        try:
            cols = [data.name]
            data = pd.DataFrame(data, columns=cols, index=idx)
        except:
            cols = ["S"]
            data = pd.DataFrame(data, columns=cols, index=idx)

    # Create uniformely distributed random phases - add the same to all time series to keep cross correlations
    if type(rand_phases) == bool:
        rand_phases = np.random.random(size=data.shape[0]) * 2 * np.pi
        try:
            rand_phases = np.reshape(np.repeat(rand_phases, data.shape[1]), data.shape)
        except IndexError:
            print(IndexError)
            pass
    # DO NOT SPECIFY AXIS ARGUMENT SINCE DATA IS SCRAMBLED AROUND IN RECONVERSION PANDAS NUMPY
    ftrafo = np.fft.fft(data)

    if ftrafo.shape != rand_phases.shape:
        rand_phases = np.reshape(rand_phases, ftrafo.shape)

    surrogate = np.multiply(ftrafo, np.exp(np.multiply(1j, rand_phases)))
    inverseftrafo = np.fft.ifft(surrogate)
    inverse_real = np.real(inverseftrafo)
    return pd.DataFrame(inverse_real, index=idx, columns=cols)


def aaft_surrogate(data):
    """
    Calculate AAFT Surrogates.
    """
    print("Calculating AAFT Surrogates")
    gaussdata = rank_remap_gaussian(data)
    surro = ft_surrogate(gaussdata)
    for idx, col in enumerate(data.columns):
        printProgressBar(idx, len(data.columns))
        surro[col] = rank_remap_TS(surro[col], data[col])
    return surro


def rank_remap_gaussian(data):
    """
    function creates a rank ordered remapping of the input data onto a
    Gaussian amplitude distribution
    """
    print("Rank ordered remapping on gaussian")
    convert_back = False

    if type(data) == pd.core.frame.DataFrame:
        convert_back = True
        cols = data.columns
        indx = data.index
        data = data.to_numpy()
    elif type(data) == list:
        data = np.array(data)

    data = data + 0.000001 * np.random.random(
        data.shape
    )  # Multiply to have random not willkürlich remapping order of same values (e.g. 0)

    remap = np.zeros(data.shape)

    try:
        dim = data.shape[1]
    except IndexError:
        dim = 1

    for j in range(dim):
        printProgressBar(j, dim - 1)

        mu = np.mean(data[:, j])
        sigma = np.std(data[:, j])
        gaussian = np.random.normal(mu, sigma, size=data.shape[0])
        gaussian = np.sort(gaussian, axis=0)
        dat = data[:, j]
        sort_dat = np.sort(data[:, j], axis=0)
        idx = [np.where(sort_dat == val)[0][0] for val in dat]
        remap[:, j] = gaussian[idx]

    if convert_back:
        remap = pd.DataFrame(remap, columns=cols, index=indx)

    return remap


def rank_remap_TS(data, target):
    """
    Creates a rank ordered remapping of the data onto the target amplitude distribution.
    Target must be 1D.
    Return value is a time series with the same amplitude distribution as the target, but the time evolution of the data.
    """
    print("Rank ordered remapping on target data")
    convert_back = False

    if type(data) == pd.core.frame.DataFrame:
        convert_back = True
        cols = data.columns
        indx = data.index
        data = data.to_numpy()
    elif type(data) == list:
        data = np.array(data).reshape((len(data), 1))

    if type(target) == pd.core.frame.DataFrame:
        target = target.to_numpy()
    elif type(target) == list:
        target = np.array(target).reshape((len(target), 1))

    data = data + 0.000001 * np.random.random(
        data.shape
    )  # Multiply to have random not willkürlich remapping order of same values (e.g. 0)
    target = target + 0.000001 * np.random.random(target.shape)

    tgt_sort = np.sort(target, axis=0)

    try:
        dim = data.shape[1]
        remap = np.zeros(data.shape)

        for j in range(dim):
            printProgressBar(j, dim - 1)
            dt_sort = np.sort(data[:, j], axis=0)
            dat = data[:, j]

            idx = [np.where(dt_sort == val)[0][0] for val in dat]
            remap[:, j] = tgt_sort[idx]

    except IndexError:
        dim = 1
        printProgressBar(1, 1)
        dt_sort = np.sort(data, axis=0)
        dat = data
        idx = [np.where(dt_sort == val)[0][0] for val in dat]
        remap = tgt_sort[idx]

    if convert_back:
        remap = pd.DataFrame(remap, columns=cols, index=indx)

    return remap


def uniDir_Maps(map_func, coupling_strength, dim, transient_steps, n_iter):
    """
    Creates a lattice of unidirectionally coupled maps where the map function is specified in map_func.
    The next iteration x^m_n+1 = map_func(e* x^m-1_n + (1-e)x^m_n) and m = 0 only depends on its past.
    transient_steps specifies the period without documentation to reach an oscillating state.or n_iter iterations.
    """
    transient = np.random.rand(dim)
    for step in range(transient_steps):
        interim = transient
        for i in range(dim):
            if i == 0:
                transient[i] = map_func(interim[i])
            else:
                transient[i] = map_func(
                    coupling_strength * interim[i - 1]
                    + (1 - coupling_strength) * interim[i]
                )
    lattice = np.array([transient])
    for step in range(n_iter):
        step_row = lattice[-1].copy()
        for i in range(dim):
            if i == 0:
                step_row[i] = map_func(lattice[-1][i])
            else:
                step_row[i] = map_func(
                    coupling_strength * lattice[-1][i - 1]
                    + (1 - coupling_strength) * lattice[-1][i]
                )
        lattice = np.append(lattice, [step_row], axis=0)
    return lattice


def tent(x):
    """
    Implementation of the tent map
    """
    if x < 0.5:
        return 2.0 * x
    else:
        return 2.0 - 2.0 * x


def ulam(x):
    return 2.0 - x * x


def pairwise(ipt, concatSign="", onSelf=False):
    """
    ipt: Input array of which a concatenation pairwise, bidirectional will be returned
    concatSign: sign to put in between pairwise concat
    onSelf: whether to include pair of self with self
    """
    output = []
    for i in ipt:
        for j in ipt:
            if i == j and onSelf == False:
                next
            else:
                if i + concatSign + j not in output:
                    output.append(i + concatSign + j)
                if j + concatSign + i not in output:
                    output.append(j + concatSign + i)
    return output


def resample(target, t_low, t_high, origin, o_low, o_high, seed=None):
    """
    NOT in Place
    Resample target data in index range [t_low: t_high] using rank ordered remapping with values from origin[o_low:o_high]
    Uniform probability distribution.
    """
    if seed:
        np.random.seed(seed)

    target = target.copy()
    resample = []
    for idx in target[t_low:t_high].index:
        resamplePos = np.random.randint(
            origin[:o_low].shape[0], origin[:o_high].shape[0]
        )
        resample.append(origin.iloc[resamplePos])

    rank_resampled = rank_remap_TS(resample, target[t_low:t_high]).flatten()
    target[t_low:t_high] = rank_resampled
    return target


def killEmptyBins(hist, bins):
    nBins = len(hist)
    for idx in np.where(hist == 0)[0]:
        if idx == 0:
            bins[1] = bins[0]
        elif idx == nBins:
            bins[-2] = bins[-1]
        else:
            newEdge = 0.5 * (bins[idx] + bins[idx + 1])
            bins[idx], bins[idx + 1] = newEdge, newEdge
    return np.sort(np.unique(bins))


def equiWidthBins(data, nBins=False, **kwargs):
    if not nBins:
        nBins = m.floor(len(data) / 10)
    return np.linspace(np.min(data), np.max(data), nBins + 1)


def sqrtNBins(data, allowEmpty=True, **kwargs):
    """
    Returns the numpy array of bin edges. Evenly spaced m.ceil(sqrt(n)) bins for n data points.
    """
    bins = np.linspace(min(data), max(data), num=int(m.ceil(m.sqrt(len(data)))))
    if not allowEmpty:
        hist, _ = np.histogram(data, bins)
        bins = killEmptyBins(hist, bins)
    return bins


def sturgesBins(data, allowEmpty=True, **kwargs):
    """
    Returns numpy array of bin edges according to Sturge's formula.
    """
    bins = np.linspace(min(data), max(data), num=int(m.ceil(m.log2(len(data))) + 1))
    if not allowEmpty:
        hist, _ = np.histogram(data, bins)
        bins = killEmptyBins(hist, bins)
    return bins


def riceRuleBins(data, allowEmpty=True, **kwargs):
    """
    Return numpy array of bin edges according to Rice's rule.
    """
    bins = np.linspace(
        min(data), max(data), num=int(m.ceil(2 * (len(data)) ** (1.0 / 3)))
    )
    if not allowEmpty:
        hist, _ = np.histogram(data, bins)
        bins = killEmptyBins(hist, bins)
    return bins


def doanesBins(data, allowEmpty=True, **kwargs):
    """
    Return numpy array of bin edges calculated by Doane's formula.
    """
    n = len(data)
    g = moment(data, moment=3)
    sg1 = m.sqrt(6 * (n - 2) / ((n + 1) * (n + 3)))
    bins = np.linspace(
        min(data), max(data), num=int(1 + m.log2(n) + m.log2(1 + abs(g) / sg1))
    )
    if not allowEmpty:
        hist, _ = np.histogram(data, bins)
        bins = killEmptyBins(hist, bins)
    return bins


def scottsRuleBins(data, allowEmpty=True, **kwargs):
    """
    Returns numpy array of bin edges calculated with Scott's rule.
    """
    step = 3.49 * np.std(data) / ((len(data)) ** (1.0 / 3))
    bins = np.arange(min(data), max(data), step=step)
    bins = np.append(bins, bins[-1] + step)
    if not allowEmpty:
        hist, _ = np.histogram(data, bins)
        bins = killEmptyBins(hist, bins)
    return bins


def freedmanDiaconisRuleBins(data, allowEmpty=True, **kwargs):
    """
    Returns numpy array of bin edges calculated with Freedman & Diaconi's rule.
    """
    step = 2 * iqr(data) / ((len(data)) ** (1.0 / 3))

    if step == 0:
        return sqrtNBins(data, allowEmpty=allowEmpty)

    bins = np.arange(min(data), max(data), step=step)
    bins = np.append(bins, bins[-1] + step)
    if not allowEmpty:
        hist, _ = np.histogram(data, bins)
        bins = killEmptyBins(hist, bins)
    return bins


def knuthBins(data, minBins=None, maxBins=None, allowEmpty=True, **_):
    """
    Creates bins according to Knuths formula. Takes some time.
    """
    if maxBins == None:
        maxBins = max(m.ceil(len(data) / 10), 3)
    if minBins == None:
        minBins = max(m.floor(len(data) / 100), 5)
    if maxBins <= minBins:
        raise ValueError(
            f"Max Bins = {maxBins}. Min bins = {minBins}. TS must be of length such that maxBins > MinBins"
        )
    logp = np.zeros(maxBins - minBins)
    N = len(data)
    for M in range(minBins, maxBins):
        n, _ = np.histogram(data, M)  # n,_,_ = plt.hist(data, M)
        part1 = N * m.log(M) + loggamma(M / 2) - loggamma(N + M / 2)
        part2 = -M * loggamma(0.5) + sum(loggamma(n + 0.5))
        logp[M - minBins] = part1 + part2
    bins = np.linspace(
        min(data), max(data), num=max(int(logp.argmax() + minBins), minBins)
    )
    if not allowEmpty:
        hist, _ = np.histogram(data, bins)
        bins = killEmptyBins(hist, bins)
    return bins


def slidingMIvsCC(data, bins, window, step):
    """
    Returns a dataframe of the temporal evolution of the mutual information and Cross Correlation
    for windows of size window and in steps of size step.
    """
    pos = 0
    res = pd.DataFrame(columns=["cc", "mi"])
    while pos + window < data.shape[0]:
        cc = cross_correlation(data.iloc[pos : pos + window, :])[0, 1]
        mi = mutual_information(data.iloc[pos : pos + window, :], bins)[0, 1]

        # Rescale MI to [-1,1] with multiplying each data point of mi with signum of cc
        mi = abs(cc) / cc * mi

        row = pd.Series([cc, mi], index=["cc", "mi"])
        row.name = data.iloc[pos].name
        res = res.append(row.copy())
        pos += step
    return res


def binEvalPlot(
    data1,
    data2,
    binFuncList,
    save=False,
    saveNamePrefix="",
    window=1000,
    step=1000,
    data1HistHeader="TS",
    data2HistHeader="Surro",
    isDateTimeIndex=True,
    sameBins=True,
):
    """
    Creates an evaluation plot for binning analysis for all binning functions in binFuncList.
    Plot shows histograms of data1 , data2 and a plot of Cross Correlation vs Mutual Infotmation
    """
    if saveNamePrefix != "":
        saveNamePrefixFileName = saveNamePrefix + "_"
        saveNamePrefix = saveNamePrefix + " "
    binDevList = []
    i = 0
    for func in binFuncList:
        printProgressBar(i, len(binFuncList))
        i += 1

        binEdges = func(data1)
        if not sameBins:
            data2BinEdges = func(data2)
            MIEdges = np.array([binEdges, data2BinEdges])
        else:
            data2BinEdges = binEdges
            MIEdges = binEdges
        ax1 = plt.subplot(221)
        n1, _, _ = ax1.hist(data1, binEdges)
        ax1.set_ylim(ymin=0, ymax=1.05 * (max(n1) + 1))
        ax1.set_title(data1HistHeader)
        ax2 = plt.subplot(222)
        n2, _, _ = ax2.hist(data2, data2BinEdges)
        ax2.set_ylim(ymin=0, ymax=1.05 * (max(n2) + 1))
        ax2.set_title(data2HistHeader)
        res = slidingMIvsCC(pd.concat([data1, data2], axis=1), MIEdges, window, step)
        res = res / res.abs().max()
        if isDateTimeIndex:
            res.index = pd.to_datetime(res.index)
        ax3 = plt.subplot(212)
        ax3.plot(res["cc"], label="cc")
        ax3.plot(res["mi"], label="mi")
        ax3.legend()
        meanAbsDev = np.mean(abs(res["cc"] - res["mi"]))
        N = len(binEdges - 1)
        ax3.set_title(f"CC vs MI mean absolute deviation: {round(meanAbsDev, 2)}")
        plt.suptitle(
            saveNamePrefix
            + func.__name__
            + f", n = {len(data1)}, N = {N}, window={window}, step = {step}"
        )
        plt.tight_layout(pad=2)
        binDevList.append((N, meanAbsDev, func.__name__))
        if save:
            plt.savefig(saveNamePrefixFileName + func.__name__ + ".pdf", dpi=300)
        plt.show()
    fig, ax = plt.subplots()

    ax.scatter([x for (x, y, name) in binDevList], [y for (x, y, name) in binDevList])
    ax.set_xlabel("N Bins")
    ax.set_ylabel("CC-Mi mean deviation")
    ax.set_title("N bins vs CCvsMI-Score, All TS")
    plt.title("N bins vs CCvsMI-Score " + saveNamePrefix)
    for i in range(len(binDevList)):
        ax.annotate(binDevList[i][2], binDevList[i][:2])
    if save:
        plt.savefig(saveNamePrefixFileName + "scores.pdf", dpi=300)
    plt.show()
    return binDevList


def uniformProbBins(data, InputPointsPerBin=None, nBins=False, **kwargs):
    """
    Creates bins of (approximately) uniform probability. If points per Bin is not a true divider of
    the length of data, excess data points are uniformly distibuted among all bins (without replacement).
    """
    n = len(data)
    if nBins:
        pointsPerBin = m.floor(n / nBins)
    elif InputPointsPerBin == None:
        pointsPerBin = m.floor(0.1 * n)
    else:
        pointsPerBin = InputPointsPerBin

    if n < pointsPerBin:
        raise Exception(
            f"Timeseries to short (length: {len(data)}) for {pointsPerBin} data points per bin. Cannot generate histogram."
        )

    remainder = n % pointsPerBin
    nBins = int((n - remainder) / pointsPerBin)

    drawToAdd = remainder % nBins
    addPerBin = int((remainder - drawToAdd) / nBins)

    # Draw bins which have more than pointsPerBin in them if pointsPerBin is not a true divider of len(data)
    plusOneBins = np.random.choice(nBins, size=drawToAdd, replace=False)

    occ = [np.count_nonzero(plusOneBins == i) for i in range(0, nBins)]
    values = np.sort(data)
    bins = [values[0]]
    pos = 0
    for i in range(0, nBins):
        pos += pointsPerBin + addPerBin + occ[i]
        if pos == n:
            bins.append(values[-1])
        else:
            bins.append(values[pos])
    return np.array(bins)


def kMeansBins(data, n_clusters=False, nBins=False, allowEmpty=True, **_):
    if not n_clusters:
        if nBins:
            n_clusters = nBins
        else:
            n_clusters = round(len(data) ** (0.5))
    kmeans = KMeans(n_clusters=n_clusters).fit(data.reshape(-1, 1))
    centers = np.sort(kmeans.cluster_centers_.flatten())
    bins = [0.5 * (centers[i + 1] + centers[i]) for i in range(len(centers) - 1)]
    bins = np.array([np.min(data)] + bins + [np.max(data)])
    if not allowEmpty:
        hist, _ = np.histogram(data, bins)
        bins = killEmptyBins(hist, bins)
    return bins


def getEpsilonFromPercentiles(data, percentile=50):
    data = np.sort(data)
    dists = []
    for i in range(1, len(data)):
        dists.append(data[i] - data[i - 1])
    return np.percentile(dists, percentile)


def getEpsilonFromNN(data, maxNN=10):
    data = np.sort(data)
    nn_dist_total_avg = []
    for i in range(1, maxNN + 1):
        mean_nn_dists = []
        for j in range(i, len(data) - i):
            nn_dists_of_j = []
            for neighbour in range(1, i + 1):
                nn_dists_of_j.append(data[j] - data[j - neighbour])
                nn_dists_of_j.append(data[j + neighbour] - data[j])
            mean_nn_dists.append(np.mean(nn_dists_of_j))
        nn_dist_total_avg.append(np.mean(mean_nn_dists))
    slope = [
        nn_dist_total_avg[i] - nn_dist_total_avg[i - 1]
        for i in range(1, len(nn_dist_total_avg))
    ]
    idx = slope.index(max(slope))
    eps = nn_dist_total_avg[idx - 1]
    return eps


def dbScanBins(ts, eps=False, allowEmpty=True, **kwargs):
    if not eps:
        # eps = getEpsilonFromPercentiles(ts)
        eps = getEpsilonFromNN(ts)
    minPts = round(0.001 * len(ts))  # More or less arbitrary as of now...
    dbscan = DBSCAN(eps=eps, min_samples=minPts).fit(ts.reshape(-1, 1))
    cluster_map = pd.DataFrame()
    cluster_map["indices"] = [i for i in range(len(ts))]
    cluster_map["cluster"] = dbscan.labels_
    cluster_map["data"] = ts
    edges = [min(cluster_map["data"])]
    for i in np.unique(cluster_map["cluster"]):
        if i == -1:
            # Noisy samples have cluster label -1, exclude for binning...
            continue
        else:
            edges.append(min(cluster_map[cluster_map["cluster"] == i]["data"]))
    edges.append(max(cluster_map["data"]))
    bins = np.array(np.sort(np.unique(edges)))
    if not allowEmpty:
        hist, _ = np.histogram(ts, bins)
        bins = killEmptyBins(hist, bins)
    return bins


def meanShiftBins(ts, allowEmpty=True, **_):
    bw = estimate_bandwidth(ts.reshape(-1, 1), quantile=0.3, n_samples=500)
    meanShift = MeanShift(bandwidth=bw).fit(ts.reshape(-1, 1))
    cluster_map = pd.DataFrame()
    cluster_map["indices"] = [i for i in range(len(ts))]
    cluster_map["cluster"] = meanShift.labels_
    cluster_map["data"] = ts
    edges = []
    for i in np.unique(cluster_map["cluster"]):
        edges.append(min(cluster_map[cluster_map["cluster"] == i]["data"]))
    edges.append(max(cluster_map["data"]))
    bins = np.array(np.sort(np.unique(edges)))
    if not allowEmpty:
        hist, _ = np.histogram(ts, bins)
        bins = killEmptyBins(hist, bins)
    return bins


def emgmmBins(ts, n_clusters=False, nBins=False, allowEmpty=True, **_):
    if not n_clusters:
        if nBins:
            n_clusters = nBins
        else:
            n_clusters = round(
                len(ts) ** 0.5
            )  # Find something to determine the best number of clusters...
    gmm = GMM(n_components=n_clusters).fit(ts.reshape(-1, 1))

    cluster_map = pd.DataFrame()
    cluster_map["indices"] = [i for i in range(len(ts))]
    cluster_map["cluster"] = gmm.predict(ts.reshape(-1, 1))
    cluster_map["data"] = ts
    edges = []
    for i in np.unique(cluster_map["cluster"]):
        edges.append(min(cluster_map[cluster_map["cluster"] == i]["data"]))
    edges.append(max(cluster_map["data"]))
    bins = np.array(np.sort(np.unique(edges)))
    if not allowEmpty:
        hist, _ = np.histogram(ts, bins)
        bins = killEmptyBins(hist, bins)
    return bins


def agglomerativeHierarchicalBins(
    ts, n_clusters=False, nBins=False, allowEmpty=True, **_
):
    if not n_clusters:
        if nBins:
            n_clusters = nBins
        else:
            n_clusters = round(len(ts) ** (0.5))
    aggloHier = AgglomerativeClustering(n_clusters=n_clusters).fit(ts.reshape(-1, 1))
    cluster_map = pd.DataFrame()
    cluster_map["indices"] = [i for i in range(len(ts))]
    cluster_map["cluster"] = aggloHier.labels_
    cluster_map["data"] = ts
    edges = []
    for i in np.unique(cluster_map["cluster"]):
        edges.append(min(cluster_map[cluster_map["cluster"] == i]["data"]))
    edges.append(max(cluster_map["data"]))
    bins = np.array(np.sort(np.unique(edges)))
    if not allowEmpty:
        hist, _ = np.histogram(ts, bins)
        bins = killEmptyBins(hist, bins)
    return bins


def minimizingCrossValidationBins(ts, nBins=False, **kwargs):
    if nBins:
        n = m.floor(nBins * 1.5)
        n_low = np.max([m.floor(nBins * 0.5), 3])
    else:
        n = len(ts)
        n_low = 3

    def j(h, p):
        return 2 / (h * (n - 1)) - (n + 1) / (h * (n - 1)) * np.sum([pj**2 for pj in p])

    best_n = n_low
    bins = np.linspace(min(ts), max(ts), best_n)
    h = bins[1] - bins[0]
    hist, _ = np.histogram(ts, bins, density=True)
    best_j = j(h, hist)
    for nBins in range(3, n_low):
        bins = np.linspace(min(ts), max(ts), nBins)
        h = bins[1] - bins[0]
        hist, _ = np.histogram(ts, bins, density=True)
        current_j = j(h, hist)
        if current_j <= best_j:  # Choose <= since more bins have higher resolution.
            best_j = current_j
            best_n = nBins
    return np.linspace(min(ts), max(ts), best_n)


def histogramEstimator(hist, bins, x):
    idx = None
    for i in range(0, len(hist)):
        if bins[i] <= x and bins[i + 1] > x:
            idx = i
            break
    if idx == None:
        return 0
    else:
        return hist[i]


def maxCrossValLikelihoodBins(ts, nBins=False, **kwargs):
    if nBins:
        n = m.floor(nBins * 1.5)
        n_low = np.max([m.floor(nBins * 0.5), 3])
    else:
        n = len(ts)
        n_low = 3

    def likelihood(hist, bins, ts):
        l = 0
        for x in ts:
            l_i = histogramEstimator(hist, bins, x)
            if l_i > 0:
                l += np.log(l_i)
        return l

    best_n = n_low
    bins = np.linspace(min(ts), max(ts), best_n)
    hist, _ = np.histogram(ts, bins, density=True)
    best_l = likelihood(hist, bins, ts)

    for nBins in range(n_low, n):
        bins = np.linspace(min(ts), max(ts), nBins)
        hist, _ = np.histogram(ts, bins)
        l = likelihood(hist, bins, ts)
        if l >= best_l:
            best_l = l
            best_n = nBins
    return np.linspace(min(ts), max(ts), best_n)


def shimazakiBins(ts, nBins=False, **kwargs):
    if nBins:
        n = m.floor(nBins * 1.5)
        n_low = np.max([m.floor(nBins * 0.5), 3])
    else:
        n = len(ts)
        n_low = 3

    def ch(hist, bins):
        h = bins[1] - bins[0]
        return (np.mean(hist) - np.std(hist)) / (n * h) ** 2

    best_n = n_low
    bins = np.linspace(min(ts), max(ts), best_n)
    hist, _ = np.histogram(ts, bins)
    best_c = ch(hist, bins)
    for nBins in range(n_low, n):
        bins = np.linspace(min(ts), max(ts), nBins)
        hist, _ = np.histogram(ts, bins)
        c = ch(hist, bins)
        if c <= best_c:
            best_c = c
            best_n = nBins
    return np.linspace(min(ts), max(ts), best_n)


def AkaikeBins(ts, nBins=False, **kwargs):
    if nBins:
        n = m.floor(nBins * 1.5)
        n_low = np.max([m.floor(nBins * 0.5), 3])
    else:
        n = len(ts)
        n_low = 3

    def AICMin(hist, bins):
        h = bins[1] - bins[0]
        nBins = len(bins) - 1
        return (
            nBins
            + n * np.log(n)
            + n * np.log(h)
            - np.sum([nj * np.log(nj) for nj in hist if nj > 0])
        )

    best_n = n_low
    bins = np.linspace(min(ts), max(ts), best_n)
    hist, _ = np.histogram(ts, bins)
    best_aic = AICMin(hist, bins)
    for nBins in range(n_low, n):
        bins = np.linspace(min(ts), max(ts), best_n)
        hist, _ = np.histogram(ts, bins)
        aic = AICMin(hist, bins)
        if aic <= best_aic:
            best_aic = aic
            best_n = nBins
    return np.linspace(min(ts), max(ts), best_n)


def smallSampleAkaikeBins(ts, nBins=False, **kwargs):
    if nBins:
        n = m.floor(nBins * 1.5)
        n_low = np.max([m.floor(nBins * 0.5), 3])
    else:
        n = len(ts)
        n_low = 3

    def AICMin(hist, bins):
        h = bins[1] - bins[0]
        nBins = len(bins) - 1
        return (
            nBins / (n / (n - nBins - 1))
            + n * np.log(n)
            + n * np.log(h)
            - np.sum([nj * np.log(nj) for nj in hist if nj > 0])
        )

    best_n = n_low
    bins = np.linspace(min(ts), max(ts), best_n)
    hist, _ = np.histogram(ts, bins)
    best_aic = AICMin(hist, bins)
    for nBins in range(n_low, n):
        bins = np.linspace(min(ts), max(ts), best_n)
        hist, _ = np.histogram(ts, bins)
        aic = AICMin(hist, bins)
        if aic <= best_aic:
            best_aic = aic
            best_n = nBins
    return np.linspace(min(ts), max(ts), best_n)


def bicBins(ts, nBins=False, **kwargs):
    if nBins:
        n = m.floor(nBins * 1.5)
        n_low = np.max([m.floor(nBins * 0.5), 3])
    else:
        n = len(ts)
        n_low = 3

    def AICMin(hist, bins):
        h = bins[1] - bins[0]
        nBins = len(bins) - 1
        return (
            np.log(n) / 2 * nBins
            + n * np.log(n)
            + n * np.log(h)
            - np.sum([nj * np.log(nj) for nj in hist if nj > 0])
        )

    best_n = n_low
    bins = np.linspace(min(ts), max(ts), best_n)
    hist, _ = np.histogram(ts, bins)
    best_aic = AICMin(hist, bins)
    for nBins in range(n_low, n):
        bins = np.linspace(min(ts), max(ts), best_n)
        hist, _ = np.histogram(ts, bins)
        aic = AICMin(hist, bins)
        if aic <= best_aic:
            best_aic = aic
            best_n = nBins
    return np.linspace(min(ts), max(ts), best_n)


def agostinoUnifProbBins(ts, **kwargs):
    n = len(ts)
    alph = 0.05

    def probit(alpha):
        return m.sqrt(2) * erfinv(2 * alpha - 1)

    nBins = 4 * (2 * n**2 / (probit(alph)) ** 2) ** (1 / 5)
    return uniformProbBins(ts, int(n / nBins))


def excludeOutliers(data, times_IQR=1.5):
    median = np.median(data)
    iq_range = iqr(data)
    upper_bound = median + times_IQR * iq_range
    lower_bound = median - times_IQR * iq_range
    return data[(data < upper_bound) & (data > lower_bound)]


def shiftBins(startPoint, endPoint, data, edges, shiftBy):
    for j in range(startPoint, endPoint):
        nextIndex = np.where(data == edges[j])[0][0] + shiftBy
        if not nextIndex >= len(data):
            nextEdge = data[nextIndex]
        else:
            nextEdge = data[-1]
        edges[j] = nextEdge
    return edges


def shapeBinsToVals(data, bins):
    for i in range(1, len(bins) - 1):
        bins[i] = min(data[data > bins[i]])
    return bins


def shapingBins(data, init=sturgesBins, allowEmpty=True):
    bins = shapeBinsToVals(data, init(data))
    hist, edges = np.histogram(data, bins)
    data = np.sort(data)
    mean = np.mean(data)
    initial_length = len(hist)
    for i in range(1, len(edges) - 1):  # len(edges) = len(hist) + 1
        meanBinRightEdge = min(edges[edges > mean])
        meanBinLeftEdge = max(edges[edges <= mean])
        binHeight = hist[i]
        priorBinHeight = hist[i - 1]

        if binHeight == -99:
            break
        if edges[i] < meanBinLeftEdge:
            if binHeight > priorBinHeight:
                continue
            else:
                shiftBy = priorBinHeight - binHeight + 1
                startPoint = i + 1
                endPoint = len(edges)
                edges = shiftBins(startPoint, endPoint, data, edges, shiftBy)
        elif edges[i] >= meanBinRightEdge:
            if binHeight < priorBinHeight:
                continue
            else:
                shiftBy = binHeight - priorBinHeight + 1
                startPoint = np.where(edges == meanBinRightEdge)[0][0]
                endPoint = i + 1
                edges = shiftBins(startPoint, endPoint, data, edges, shiftBy)
        else:  # Is mean Bin
            try:
                nextBinHeight = hist[i + 1]
            except IndexError:
                continue
            if binHeight > priorBinHeight and binHeight > nextBinHeight:
                continue
            elif binHeight < priorBinHeight and binHeight < nextBinHeight:
                shiftBy = max(
                    priorBinHeight - binHeight + 1, nextBinHeight - binHeight + 1
                )
                startPoint = i + 1
                endPoint = len(edges)
                edges = shiftBins(startPoint, endPoint, data, edges, shiftBy)
            elif binHeight < priorBinHeight:
                shiftBy = priorBinHeight - binHeight + 1
                startPoint = i + 1
                endPoint = len(edges)
                edges = shiftBins(startPoint, endPoint, data, edges, shiftBy)
            elif binHeight < nextBinHeight:
                shiftBy = nextBinHeight - binHeight
                startPoint = np.where(edges == meanBinRightEdge)[0][0]
                endPoint = np.where(edges == meanBinRightEdge)[0][0] + 1
                edges = shiftBins(startPoint, endPoint, data, edges, shiftBy)

        if edges[i] == data[-1]:
            break
        hist, _ = np.histogram(data, edges)
        for i in range(initial_length - len(hist)):
            hist = np.append(hist, [-99])

    bins = np.sort(np.unique(edges))
    if not allowEmpty:
        hist, _ = np.histogram(data, bins)
        bins = killEmptyBins(hist, bins)
    return bins


def backwardBins(data, allowEmpty=True):
    # TODO implement check if one value more times in bin and dist skewed such that not infinitely running ...
    hist = np.array([len(data)])
    bins = np.array([min(data), max(data)])
    prior_bins = bins
    while min(hist) > m.ceil(0.05 * len(data)):
        prior_bins = bins
        hist, bins = np.histogram(data, np.array([min(data), max(data)]))
        biggestBinIdx = np.where(hist == max(hist))[0][0]
        newEdge = +0.5 * (bins[biggestBinIdx + 1] + bins[biggestBinIdx])
        bins = np.append(
            bins[: biggestBinIdx + 1], [np.array([newEdge]), bins[biggestBinIdx + 1 :]]
        )
        if np.array_equal(bins, prior_bins):
            break
    if not allowEmpty:
        bins = killEmptyBins(hist, bins)
    return bins


def deviations2Cost(nBins, nData, deviationsArray):
    deviationsArray = deviationsArray.reshape(-1, 2)
    deviations = np.array([entry[0] - entry[1] for entry in deviationsArray])
    return 1 / nData * np.sqrt(sum(deviations**2))


# Factor n to penalize number of bins. Divide by nData to unpenalize longer TimeSeries
def NDeviations2Cost(nBins, nData, deviationsArray):
    deviationsArray = deviationsArray.reshape(-1, 2)
    deviations = np.array([entry[0] - entry[1] for entry in deviationsArray])
    return nBins / nData * np.sqrt(sum(deviations**2))


def NDeviationsAbsCost(nBins, nData, deviationsArray):
    deviationsArray = deviationsArray.reshape(-1, 2)
    deviations = np.array([entry[0] - entry[1] for entry in deviationsArray])
    return nBins / nData * np.sqrt(sum(abs(deviations)))


def N2Deviations2Cost(nBins, nData, deviationsArray):
    deviationsArray = deviationsArray.reshape(-1, 2)
    deviations = np.array([entry[0] - entry[1] for entry in deviationsArray])
    return nBins**2 / nData * np.sqrt(sum(deviations**2))


def avgLogLikelihood(nBins, nData, deviationsArray):
    # Problem: gives zero error if original pdf(x) = 0 even when est_pdf(x) != 0
    deviationsArray = deviationsArray.reshape(-1, 2)
    deviations = np.array(
        [
            entry[1] * np.log(abs(entry[1] / entry[0]))
            if entry[0] * entry[1] != 0
            else entry[1]
            * np.log(abs(entry[1] / (entry[0] + 0.001 * np.random.normal())))
            if entry[1] != 0
            else 0
            for entry in deviationsArray
        ]
    )
    return nBins / nData * sum(abs(deviations))


def evaluateBins(data, bins, pdf, costfunction):
    hist, _ = np.histogram(data, bins, density=True)
    deviations = []

    # Calculate deviations for data points of ts
    for i in range(len(hist) - 1):
        bin_content = data[(data >= bins[i]) & (data < bins[i + 1])]
        for j in range(len(bin_content)):
            deviations.append((hist[i], pdf(bin_content[j])))

    # calculate deviations for random points in range. sample 10% of ts length but at least 100 data points
    span = abs(max(data) - min(data))
    randomTestSample = np.random.uniform(
        low=min(data) - 0.25 * span,
        high=max(data) + 1.25 * span,
        size=max(100, m.ceil(0.1 * len(data))),
    )

    hist2, _ = np.histogram(data, bins, density=True)
    for i in range(len(hist2) - 1):
        bin_content = randomTestSample[
            (randomTestSample >= bins[i]) & (randomTestSample < bins[i + 1])
        ]
        for j in range(len(bin_content)):
            deviations.append((hist2[i], pdf(bin_content[j])))

    deviations = np.array(deviations)
    return costfunction(len(hist), len(data) + len(randomTestSample), deviations)


def evaluateKDEs(data, KDE, pdf, costfunction):
    span = abs(max(data) - min(data))
    randomTestSample = np.random.uniform(
        low=min(data) - 0.25 * span,
        high=max(data) + 1.25 * span,
        size=max(100, m.ceil(0.1 * len(data))),
    )

    testData = np.append(data, randomTestSample)
    deviations = []
    estimated = KDE(testData.reshape(-1, 1)).flatten()
    for i in range(len(testData)):
        deviations.append((estimated[i], pdf(testData[i])))
    deviations = np.array(deviations)
    cost = costfunction(1, len(testData), deviations)
    return cost


def power(x, a=2):
    return a * x ** (a - 1.0)


def pareto(x, a=2.3, m=1):
    if x < m:
        return 0
    else:
        return a * x ** (-a - 1) * m**a


def inversePareto(x, a=2):
    return (a / x) ** (1 / (a + 1))


def standardNormal(x, sigma=1, mu=0, **kwargs):
    return 1 / (sigma * np.sqrt(m.pi * 2)) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def evaluate(
    dist, func, binFuncs, costFuncs, n_iter, resultsDf, plot=True, *args, **kwargs
):
    ts_array = [dist(*args, **kwargs) for i in range(n_iter)]
    interimDf = pd.DataFrame(columns=resultsDf.columns)
    for estFunc in binFuncs:
        if "KDE" in estFunc.__name__:
            isKDE = True
            isBin = False
            fitDists = np.array([estFunc(ts) for ts in ts_array])
        elif "Bins" in estFunc.__name__:
            isKDE = False
            isBin = True
            binsList = np.array([estFunc(ts) for ts in ts_array])
        for cost in costFuncs:
            if isKDE:
                deviations = np.array(
                    [
                        evaluateKDEs(ts_array[idx], fitDists[idx], func, cost)
                        for idx in range(len(ts_array))
                    ]
                )
            elif isBin:
                deviations = np.array(
                    [
                        evaluateBins(ts_array[idx], binsList[idx], func, cost)
                        for idx in range(len(ts_array))
                    ]
                )
            if plot:
                for i in range(len(ts_array)):
                    plt.hist(ts_array[i], binsList[i], density=True, alpha=0.1)
                totalMin = min([min(ts) for ts in ts_array])
                totalMax = max([max(ts) for ts in ts_array])
                xVals = np.arange(1.1 * totalMin, 1.1 * totalMax, step=0.01)
                plt.plot(xVals, [func(x) for x in xVals])
                plt.title(
                    f"{estFunc.__name__}: mean = {np.mean(deviations):.2e}, std = {np.std(deviations):.2e}"
                )
                plt.show()
            interimDf["devs"] = deviations
            interimDf["func"] = estFunc.__name__
            interimDf["dist"] = func.__name__
            interimDf["len"] = kwargs["size"]
            interimDf["cost"] = cost.__name__
            # print('{}: mean = {:.3e}, std = {:.3e}'.format(estFunc.__name__,np.mean(deviations), np.std(deviations)) )
            resultsDf = resultsDf.append(interimDf, ignore_index=True).copy()
    return resultsDf


def kdeProb(X, kd):
    if type(X) == int or type(X) == float or type(X) == np.float_:
        X = np.array([[X]])
    elif type(X[0]) == float or type(X[0]) == int or type(X[0]) == np.float_:
        X = np.array([X]).reshape(-1, 1)
    return np.exp(kd.score_samples(X)).flatten()


def gaussianKDE(ts):
    bw = estimate_bandwidth(ts.reshape(-1, 1), quantile=0.3, n_samples=500)
    kd = KD(kernel="gaussian", bandwidth=bw).fit(ts.reshape(-1, 1))
    return lambda X: kdeProb(X, kd)


def tophatKDE(ts):
    bw = estimate_bandwidth(ts.reshape(-1, 1), quantile=0.3, n_samples=500)
    kd = KD(kernel="tophat", bandwidth=bw).fit(ts.reshape(-1, 1))
    return lambda X: kdeProb(X, kd)


def epanechnikovKDE(ts):
    bw = estimate_bandwidth(ts.reshape(-1, 1), quantile=0.3, n_samples=500)
    kd = KD(kernel="epanechnikov", bandwidth=bw).fit(ts.reshape(-1, 1))
    return lambda X: kdeProb(X, kd)


def expKDE(ts):
    bw = estimate_bandwidth(ts.reshape(-1, 1), quantile=0.3, n_samples=500)
    kd = KD(kernel="exponential", bandwidth=bw).fit(ts.reshape(-1, 1))
    return lambda X: kdeProb(X, kd)


def linearKDE(ts):
    bw = estimate_bandwidth(ts.reshape(-1, 1), quantile=0.3, n_samples=500)
    kd = KD(kernel="linear", bandwidth=bw).fit(ts.reshape(-1, 1))
    return lambda X: kdeProb(X, kd)


def cosineKDE(ts):
    bw = estimate_bandwidth(ts.reshape(-1, 1), quantile=0.3, n_samples=500)
    kd = KD(kernel="cosine", bandwidth=bw).fit(ts.reshape(-1, 1))
    return lambda X: kdeProb(X, kd)


def areaBetweeenHistFunc(hist_density, bins, func, intLimit=1000, **kwargs):
    def histFunc(x):
        if x < min(bins) or x >= max(bins):
            return 0
        else:
            idx = max(np.where(bins <= x)[0])
            return hist_density[idx]

    def distAtX(x):
        return abs(histFunc(x) - func(x, **kwargs))

    return integrate.quad(distAtX, -np.inf, np.inf, limit=intLimit)[0]


def evaluateHistFunc(
    hist, bins, func, numTestPoints=5000, a=-1.1, b=1.1, hist_is_density=True, **kwargs
):
    def histFunc(x):
        if x < min(bins) or x >= max(bins):
            return 0
        else:
            idx = max(np.where(bins <= x)[0])
            if hist_is_density:
                return hist[idx]
            else:
                return hist[idx] / np.sum(hist)

    testData = np.linspace(a, b, numTestPoints)

    return (
        1
        / numTestPoints
        * np.sum([np.sqrt((histFunc(x) - func(x)) ** 2) for x in testData])
    )


def areaBetweenKDEFunc(KDE, func, intLimit=1000, **kwargs):
    def distAtX(x):
        xArray = np.array([[x]])
        return abs(KDE(xArray) - func(x, **kwargs))

    return integrate.quad(distAtX, -np.inf, np.inf, limit=intLimit)[0]


def iterativeMaxEntBins(
    data, nBins=None, n_iter=50, steps_per_bin_pair=50, randomizeShifts=True, **_
):
    if nBins == None:
        nBins = int(len(data) / 10)
    sData = np.sort(data)
    span = sData[-1] - sData[0]
    N = len(data)

    bins = np.linspace(sData[0], sData[-1], num=nBins + 1)

    def cost(histVals, bins):
        Nb = len(bins) - 1  # Exclude right edge fron N_edges to get N_bins
        unif = 1.0 / Nb

        sigma = 0
        # do not draw sqrt bc. of rounding err.
        for i in range(1, len(bins)):
            w_i = bins[i] - bins[i - 1]
            sigma += (w_i / span - unif) ** 2
        for i in range(len(histVals)):
            sigma += (histVals[i] / N - unif) ** 2
        return sigma

    hist, _ = np.histogram(data, bins)
    cost0 = cost(hist, bins)
    for it in range(n_iter):
        printProgressBar(it, n_iter)
        bins0 = bins.copy()
        idxs = np.arange(1, len(hist) - 1)
        if randomizeShifts:
            np.random.shuffle(idxs)
        for idx in idxs:
            bin_steps = np.linspace(
                bins[idx - 1], bins[idx + 1], num=steps_per_bin_pair
            )

            for jdx in range(1, len(bin_steps) - 1):
                bins_i = bins.copy()
                bins_i[idx] = bin_steps[jdx]
                hist_i, bins_i = np.histogram(data, bins_i)
                cost_i = cost(hist_i, bins_i)
                if cost_i < cost0:
                    cost0 = cost_i
                    bins = bins_i
                    hist = hist_i
        comp = bins == bins0
        if comp.all():
            printProgressBar(n_iter, n_iter)
            break
    return bins


def normal_dist(x, mean=0, sd=1):
    prob_density = 1 / np.sqrt(2 * np.pi) * sd * np.exp(-1 / 2 * ((x - mean) / sd) ** 2)
    return prob_density


def laplace_dist(x, m=0, b=1):
    return 1 / (2 * b) * np.exp(-abs(x - m) / b)


def exponential_dist(x, beta=1):
    if x < 0:
        return 0
    else:
        return 1 / beta * np.exp(-x / beta)


def uniform_dist(x, low=0, high=1):
    if low <= x and high >= x:
        return 1 / (high - low)
    else:
        return 0


def logistic_dist(x, m=0, s=1):
    return np.exp(-(x - m) / s) / (s * (1 + np.exp(-(x - m) / s)) ** 2)


def gumbel_dist(x, m=0, s=1):
    z = (x - m) / s
    return 1 / s * np.exp(-(z + np.exp(-z)))


def pareto_dist(x, m=1, a=2.3):
    return a * m**a / x ** (a + 1)


def beta_dist(x, a=0.5, b=0.5):
    return beta.pdf(x, a, b)


def argus_dist(x, chi=1, loc=0):
    return argus.pdf(x, chi, loc)


def r_dist(x, c=1.6):
    return rdist.pdf(x, c)


def rescale(data, minPt=0, maxPt=1):
    rescaled_data = data.copy()
    if len(data.shape) == 1:
        if (rescaled_data.max() - rescaled_data.min()) == 0:
            return rescaled_data
        else:
            return (rescaled_data - rescaled_data.min()) / (
                rescaled_data.max() - rescaled_data.min()
            ) * (maxPt - minPt) + minPt
    else:
        for col in range(rescaled_data.shape[1]):
            rescaled_data[:, col] = (
                rescaled_data[:, col] - rescaled_data[:, col].min()
            ) / (rescaled_data[:, col].max() - rescaled_data[:, col].min()) * (
                maxPt - minPt
            ) + minPt
        return rescaled_data


def link_density(adjMtx):
    return np.sum(adjMtx) / (adjMtx.shape[0] * (adjMtx.shape[0] - 1))
