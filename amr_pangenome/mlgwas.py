#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 3 18:49:02 2020

@author: jhyun95

"""

import collections
import numpy as np
import pandas as pd
import scipy.stats

def gwas_rse(df_features, df_labels, null_shuffle=False, base_model=None):
    '''
    Runs a random subspace ensemble using sklearn BaggingClassifier.
    Can specify a base_model, or will use an L1-normalized LinearSVC by default.
    
    Parameters
    ----------
    df_features : pd.DataFrame
        Binary samples x features DataFrame. Assumes all values are 0 or 1.
        Will automatically transpose if necessary.
    df_labels : pd.Series
        Binary Series corresponding to samples in df_features. 
    null_shuffle : bool
        If True, randomly shuffles labels for null model testing (default False)
    base_model : sklearn classifier
        Any sklearn classifier
    '''
    
    ''' Shuffle labels for null model '''
    X, y = setup_Xy(df_features, df_labels, null_shuffle)
    
    ''' Set up classifier '''
    #TODO
    

def gwas_fisher_exact(df_features, df_labels, null_shuffle=False, report_rate=10000):
    ''' 
    Applies Fisher Exact test between each feature the label. 
    Returns a dataframe with TPs/FPs/FNs/TNs/signed pvalues/oddsratios
    Adapted from scripts/microbial_gwas.py 
    
    Parameters
    ----------
    df_features : pd.DataFrame
        Binary samples x features DataFrame. Assumes all values are 0 or 1.
        Will automatically transpose if necessary.
    df_labels : pd.Series
        Binary Series corresponding to samples in df_features. 
    null_shuffle : bool
        If True, randomly shuffles labels for null model testing (default False)
    report_rate : int
        Prints message each time this many features are processed (default 10000)
    '''
    
    def __batch_contingency__(X, y):
        ''' Creates a 3D table of contingency tables such that
            Table[i] = [[a,b],[c,d]] = contigency table for ith entry
                     = [[TP,FP],[FN,TN]] '''
        Y = np.broadcast_to(y, reversed(X.shape)).T
        Xb = X.astype(bool); Yb = Y.astype(bool)
        contingency = np.zeros(shape=(Xb.shape[1],2,2))
        contingency[:,0,0] = np.sum(np.logical_and(Xb,Yb), axis=0) # true positives
        contingency[:,0,1] = np.sum(np.logical_and(Xb,1-Yb), axis=0) # false positives
        contingency[:,1,0] = np.sum(np.logical_and(1-Xb,Yb), axis=0) # false negatives
        contingency[:,1,1] = np.sum(np.logical_and(1-Xb,1-Yb), axis=0) # true negatives
        return contingency
        
    ''' Shuffle labels for null model '''
    X, y = setup_Xy(df_features, df_labels, null_shuffle)
    
    ''' Compute contingency tables per feature '''
    contingency = __batch_contingency__(X, y)
    print 'Computed contingency tables:', contingency.shape
    
    ''' Apply Fisher's exact tests and compute ORs per feature '''
    df_output = pd.DataFrame(index=df_features.index if df_features.shape[1] == y.shape[0] else df_features.columns,
                     columns=['TPs','FPs','FNs','TNs'], data=contingency.reshape(-1,4))
    pvalues = np.zeros(X.shape[1])
    oddsratios = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        if (i % report_rate) == 0:
            print 'Feature', i
        oddsratio, pvalue = scipy.stats.fisher_exact(contingency[i,:,:])
        pvalue = -pvalue if oddsratio < 1 else pvalue # signed pvalue by direction
        pvalues[i] = pvalue
        oddsratios[i] = oddsratio
    df_output['pvalue'] = pvalues
    df_output['oddsratio'] = oddsratios
    return df_output


def binarize_amr_table(df_amr):
    ''' 
    Converts SIR values in a pandas DataFrame or Series into binary phenotypes 
        - Susceptible (0): "Susceptible", "Susceptible-dose dependent" 
        - Non-Susceptible (1): "Resistant", "Intermediate", "Non-susceptible"
        - Missing (np.nan): np.nan, "Not defined", anything else
        
    Parameters 
    ----------
    df_amr : pd.DataFrame or pd.Series
        Dataframe with raw SIR values
        
    Returns
    -------
    df_pheno : pd.DataFrame or pd.Series
        Dataframe with binarized SIR values
    sir_counts : dict
        Dictionary mapping counts of original SIRs (for Series), or nested
        dictionary mapping drug:original SIR:count (for DataFrame)
        
    '''
    
    def binarize_sir(sir):
        ''' Binarizes a single SIR value '''
        if sir in ['Susceptible', 'Susceptible-dose dependent']:
            return 0
        elif sir in ['Resistant', 'Intermediate', 'Non-susceptible']:
            return 1
        elif sir in [np.nan, 'Not defined']:
            return np.nan
        else:
            print 'Unknown SIR:', sir
            return np.nan
        
    ''' Process input column-wise '''
    if type(df_amr) == pd.Series: # single column
        sir_counts = dict(collections.Counter(df_amr.values))
        df_pheno = df_amr.map(binarize_sir)
        return df_pheno, sir_counts
    elif type(df_amr) == pd.DataFrame: # process DataFrame column-by-column
        processed = map(lambda col: process_amr_table(df_amr[col]), df_amr.columns)
        processed_columns, sir_counters = zip(*processed)
        sir_counts = {df_amr.columns[i]:sir_counters[i] for i in range(df_amr.shape[1])}
        df_pheno = pd.concat(processed_columns, axis=1)
        return df_pheno, sir_counts
    else: # unknown format
        print 'Input not Series or DataFrame'
        return df_amr, None

    
def setup_Xy(df_features, df_labels, null_shuffle=False):
    ''' Takes feature DataFrame and label Series and returns the 
        X and y numpy arrays to use with sklearn. Takes into account
        shuffling for null models, and potential need to transpose. '''
    X = df_features.values; y = df_labels.values; n = y.shape[0]
    if X.shape[0] != n: # dimension mismatch
        if X.shape[1] == n: # transpose provided
            X = X.T
        else:
            print 'Incompatible dimensions:', df_features.shape, df_labels.shape
            return None
    if null_shuffle:
        y = np.random.shuffle(y)
    return X, y


def filter_nonvariable(df_features, min_variation=2):
    ''' 
    Removes features that have very low variability. For example,
    if min_variation=3, removes features that present in only 1 or 2
    features, or features missing in only 0, 1, or 2 features. 
    Creates a copy of df_features, intended to speed up ML. 
    
    Parameters
    ----------
    df_features : pd.DataFrame
        Feature matrix as features (rows) x samples (cols)
    min_variation : int
        Minimum variation to keep a feature, see explanation above (default 2)
        
    Returns
    -------
    df : pd.DataFrame
        Copy of df_features with low variation features removed
    '''
    df_counts = df_features.sum(axis=1)
    num_features, num_samples = df_features.shape
    min_count = min_variation
    max_count = num_samples - min_variation # inclusive count bounds
    selected_features = df_counts[(df_counts >= min_count) & (df_counts <= max_count)].index
    return df_features.loc[selected_features,:]