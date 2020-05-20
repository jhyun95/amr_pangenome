#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed May 6 16:35:52 2020

@author: jhyun95
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.cluster.hierarchy as sch

from mlgwas import sparse_arrays_to_sparse_matrix


def fit_heaps_law(df_pan_core, drop_early=0, figsize=(8,4)):
    '''
    Esimates pan-genome openness by fitting Heap's Law between the 
    new gene rate and the number of genomes. Based on median pan-genome
    sizes computed by estimate_pan_core_size(). Fits NGR = k*(genomes)^-a
    
    Parameters
    ----------
    df_pan_core : pd.DataFrame
        Table of pan/core genome sizes from estimate_pan_core_size()
    drop_early : int
        If positive, excludes this many initial points in the 
        pan-genome curve when fitting Heap's Law. Can give smoother
        overall fits. Ideally not more than 2 (default 0)
    figsize : tuple
        Width, height in inches, from mpl.figure (default (8,4))
        
    Returns
    -------
    k : float
        k parameter from fitted Heaps' Law, scaling factor
    a : float
        a parameter from fitted Heaps' Law, exponent or "openness",
        where closer to 0 = more open.
    axs : (mpl.axes, mpl.axes)
        Pair of Axes corresponding to pan/core and Heaps' Law plots.
    '''
    
    ''' Compute median pan and core genome sizes '''
    num_iter, num_cols = df_pan_core.shape
    num_strains = num_cols / 2
    median_pan_genome = np.median(df_pan_core.iloc[:,:num_strains].values, axis=0)
    median_core_genome = np.median(df_pan_core.iloc[:,-num_strains:].values, axis=0)
    
    ''' Select points to fit '''
    new_gene_rates = median_pan_genome[1:] - median_pan_genome[:-1] 
    new_gene_rates = np.insert(new_gene_rates, 0, median_pan_genome[0])
    end_limit = num_strains
    while new_gene_rates[end_limit-1] == 0: # trim cases where 0 new genes/strain are added to avoid NGR=0
        end_limit -= 1
    included_strain_counts = range(drop_early+1, end_limit+1)
    included_gene_rates = new_gene_rates[drop_early:end_limit]
    
    ''' Fit Heaps' Law to selected points '''
    heaps_law = lambda N, k, a: k * np.power(N,-a) # y = k*N^-a; log(y) = log(k) - a*log(N)
    x = np.log(included_strain_counts) # log strain counts, or log(N)
    y = np.log(included_gene_rates) # log new gene rates, or log(NGR) = log(y)
    A = np.vstack([x, np.ones(len(x))]).transpose() # formulate as linear regression
    neg_a, logk = np.linalg.lstsq(A, y)[0]
    a = -neg_a; k = np.exp(logk)
    print "Heap's Law (k*N^-a): k =", k, '| a =', a
    fitted_new_gene_rates = heaps_law(included_strain_counts, k, a)
    
    ''' Generate pan/core genome plot and Heap's Law plot '''
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    strain_counter = range(1,num_strains+1)
    axs[0].plot(strain_counter, median_pan_genome, label='Pan-genome')
    axs[0].plot(strain_counter, median_core_genome, label='Core-genome')
    axs[0].set_xlabel('num. strains'); axs[0].set_ylabel('num. genes')
    axs[0].legend(); axs[0].set_ylim(bottom=0)
    axs[1].loglog(included_strain_counts, included_gene_rates, label='Median NGR')
    axs[1].loglog(included_strain_counts, fitted_new_gene_rates, label="Heaps' Law (fit)")
    axs[1].legend()
    axs[1].set_xlabel('num. strains'); axs[1].set_ylabel('num. new genes')
    plt.tight_layout()
    return k, a, axs


def estimate_pan_core_size(df_genes, num_iter=1000, log_batch=100):
    '''
    Estimates pan- and core-genome size curves by randomly shuffling
    strain order and computing the sizes as strains are added one-by-one.
    
    Parameters
    ----------
    df_genes : pd.DataFrame
        Binary gene x strain table.
    num_iter : int
        Number of shuffling iterations to do
    log_batch : int
        Report progress every log_batch iterations. Does not
        report anything if negative (default 100).
    
    Returns
    -------
    df_pan_core : pd.DataFrame
        Given N strains, returns a DataFrame where rows are iterations, 
        the first N columns are pan-genome sizes for 1 to N shuffled 
        strains, and the next N columns are corresponding core-genome sizes.
    '''
    num_genes, num_strains = df_genes.shape
    pan_genomes = np.zeros((num_iter, num_strains)) # estimated pan-genome curve per iteration
    core_genomes = np.zeros((num_iter, num_strains)) # estimated core-genome curve per iteration
    gene_data = df_genes.fillna(0).values.T # now strain x cluster
    
    ''' Simulate pan/core-genomes for randomly ordered strains '''
    for i in range(num_iter):
        if log_batch > 0 and (i+1) % log_batch == 0: 
            print 'Shuffling', i+1, 'of', num_iter
        np.random.shuffle(gene_data) # shuffles rows = strains
        pan_genomes[i,:] = np.cumsum(gene_data, axis=0).astype(bool).sum(axis=1) 
            # [i,j] entry = size of pangenome with j genomes, for iteration i
        core_genomes[i,:] = np.cumprod(gene_data, axis=0).astype(bool).sum(axis=1) 
            # [i,j] entry = size of core genome with j genomes, for iteration i
    
    ''' Save to DataFrame '''
    iter_index = map(lambda x: 'Iter' + str(x), range(1,num_iter+1))
    pan_cols = map(lambda x: 'Pan' + str(x), range(1,num_strains+1))
    core_cols = map(lambda x: 'Core' + str(x), range(1,num_strains+1))
    df_pan_core = pd.DataFrame(index=iter_index, columns=pan_cols + core_cols,
                               data=np.hstack([pan_genomes, core_genomes]))
    return df_pan_core


def plot_strain_hierarchy(df_features, df_labels=None, colors={1:'red', 0:'blue'}, 
    linkage='average', figsize=(8,6), polar=False):
    ''' 
    Cluster strains by Jaccard distance (expects binary gene x strain table),
    and plots the hierarchy. Can optionally color by label.
    
    Parameters
    ----------
    df_features : pd.DataFrame
        Binary feature x strain table, used to compute pairwise Jaccard distances.
        Optimized for dataframes with SparseArray columns.
    df_labels : pd.Series
        Label per strain, optionally for color-coding text labels (default None)
    colors : dict
        Maps labels to colors, only used when df_labels is not None (default None)
    linkage : str
        Linkage method to use with scipy.cluster.hierarchy.linkage() (default 'average')
    figsize : tuple
        Hierarchy figure dimensions
    polar : bool
        If True, creates a polar dendrogram (default False)
    '''
    num_features, num_strains = df_features.shape
    strains = df_features.columns

    ''' Compute pairwise Jaccard distances '''
    print 'Computing pairwise Jaccard distances...'
    is_sparse = reduce(lambda x,y: x and y, map(lambda x: 'sparse' in x, df_features.ftypes))
    if is_sparse: # columns are SparseArrays
        spdata = sparse_arrays_to_sparse_matrix(df_features).tocsc() # df -> COO -> CSC
        distances = __fast_pairwise_jaccard__(spdata)
    else: # columns are dense
        distances = _fast_pairwise_jaccard__(df_features.values)

    ''' Generate dendrogram '''
    dist_condensed = distances[np.triu_indices(num_strains,1)]
    Z = sch.linkage(dist_condensed, linkage)
    is_colorable = not (df_labels is None or colors is None)
    df_colors = df_labels.map(colors) if is_colorable else None

    if polar: 
        dend = sch.dendrogram(Z, labels=strains, no_plot=True)
        plot_polar_dendogram(dend, figsize=figsize, df_colors=df_colors)
    else: # TODO: Colored text labels for non-polar dendrogram
        fig, ax = plt.subplots(1,1, figsize=figsize)
        dend = sch.dendrogram(Z, labels=strains, ax=ax)
    df_distances = pd.DataFrame(index=strains, columns=strains, data=distances)
    return dend, df_distances


def plot_polar_dendogram(dend, figsize=(8,8), df_colors=None):
    ''' 
    Use with plot_strain_hierarchy().
    Adapted from https://stackoverflow.com/questions/51936574/
    how-to-plot-scipy-hierarchy-dendrogram-using-polar-coordinates
    '''
    icoord = np.array(dend['icoord'])
    dcoord = np.array(dend['dcoord'])
    
    ''' Transformations for polar coordinates + formatting '''
    def smooth_segment(seg, Nsmooth=100):
        return np.concatenate([[seg[0]], np.linspace(seg[1], seg[2], Nsmooth), [seg[3]]])

    gap = 0.1
    dcoord = -dcoord
    #dcoord = -np.log(dcoord+1) # log transform, sometimes makes polar dendrogram look better
    imax = icoord.max()
    imin = icoord.min()
    icoord = ((icoord - imin)/(imax - imin)*(1-gap) + gap/2)*2*np.pi
    
    ''' Plotting polar dendrogram '''
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, polar=True)
    for xs,ys in zip(icoord, dcoord):
        xs = smooth_segment(xs)
        ys = smooth_segment(ys)
        ax.plot(xs, ys, color="black")
    
    ''' Adjust polar tick labels '''
    ax.spines['polar'].set_visible(False)
    ax.set_rlabel_position(0)
    num_xticks = dcoord.shape[0]+1
    angles = np.linspace(gap/2, 1-gap/2, num_xticks) * np.pi * 2
    ax.set_xticks(angles) #*np.pi*2)
    ax.set_xticklabels(dend['ivl'])

    plt.gcf().canvas.draw()
    for label, angle in zip(ax.get_xticklabels(), angles):
        x,y = label.get_position()
        label_text = label.get_text()
        shift = 0.007 * len(label_text)
        if not df_colors is None and label_text in df_colors.index:
            color = df_colors.loc[label_text]
        else:
            color = 'black'
        color = color if pd.notnull(color) else 'black'
        lab = ax.text(x,y-shift, label.get_text(), transform=label.get_transform(),
            ha=label.get_ha(), va=label.get_va(), color=color)
        
        if angle > 0.5*np.pi and angle < 1.5*np.pi:
            angle += np.pi
        lab.set_rotation(angle * 180.0 / np.pi)

    ax.set_xticklabels([])
    return fig, ax


def __fast_pairwise_jaccard__(mat):
    ''' 
    Computes pairwise Jaccard distances between columns by tiling columns for
    multiple comparisons. For example, when comparing col 1 vs 2,3,4,...10,
    compares [c1,c1,...,c1] vs [c2,c3,...,c10], rather than setting up individual
    c1 vs c2, c1 vs c3, ..., c1 vs c10 comparisons. Works for numpy arrays and 
    scipy.sparse matrices that support column slicing (i.e. CSC).
    '''
    is_sparse = scipy.sparse.issparse(mat)
    tiler = scipy.sparse.hstack if is_sparse else np.hstack
    nrows, ncols = mat.shape
    distances = np.zeros((ncols,ncols))
    for c in range(1,ncols):
        col1 = mat[:,c].astype('bool')
        col1_tiled = tiler([col1] * c) 
        col2 = mat[:,:c].astype('bool')
        shared = (col1_tiled.multiply(col2)).sum(axis=0) # multiply = AND
        union = (col1_tiled + col2).sum(axis=0) # add = OR
        dist = 1.0 - (np.divide(shared, union.astype('float')))
        distances[c,:c] = dist
        distances[:c,c] = dist
    return distances