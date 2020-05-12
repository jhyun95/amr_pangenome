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
        spdata = __sparse_arrays_to_sparse_matrix__(df_features).tocsc() # df -> COO -> CSC
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


def __sparse_arrays_to_sparse_matrix__(dfs):
    '''
    Converts a binary DataFrame with SparseArray columns into a
    scipy.sparse.coo_matrix.
    '''
    num_entries = dfs.fillna(0).sum().sum()
    positions = np.empty((2,num_entries), dtype='int')
    fill_values = np.empty(num_entries)
    current = 0
    for i in range(dfs.shape[1]):
        col_entries = dfs.iloc[:,i].values
        col_num_entries = col_entries.sp_index.npoints
        positions[0, current:current+col_num_entries] = col_entries.sp_index.indices
        positions[1, current:current+col_num_entries] = i
        fill_values[current:current+col_num_entries] = col_entries.sp_values
        current += col_num_entries
    spdata = scipy.sparse.coo_matrix((fill_values, positions), shape=dfs.shape)
    return spdata


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