#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Oct Apr 7 12:42:39 2021

@author: jhyun95

Miscellaneous ML optimization pipelines for AMR-ML 2.0.
"""

from __future__ import print_function
import time
import numpy as np
import pandas as pd
import scipy.stats
import sklearn.model_selection, sklearn.metrics
import amr_pangenome.sparse_utils

PROJECT_DIR = '/media/pekar2/AMR_MLGWAS/'

def evaluate_model(base_clf, lsdf_case, case_block_defs, df_amr_org_drug,
                   known_amr_drug_set, n_folds=5, seed=np.random.rand()):
    '''
    Evaluates a classifiers's prediction performance and recovery of known AMR
    genes with stratified cross validation.
    '''
    X = lsdf_case.data.T.tocsr()
    y = df_amr_org_drug.values.astype(int)
    output = {}; fold = 1
    skf = sklearn.model_selection.StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=seed)
    
    ''' Cross validation loop '''
    for train_index, test_index in skf.split(X, y):
        start_time = time.time()
        print('FOLD:', fold)
        fold_id = 'FOLD' + str(fold); fold += 1
        output[fold_id] = {}

        ''' Train model '''
        print('\tTraining model...')
        X_train, X_test = X[train_index,:], X[test_index,:]
        y_train, y_test = y[train_index], y[test_index]
        clf = sklearn.base.clone(base_clf)
        clf.fit(X_train, y_train)
        
        ''' Evaluate prediction '''
        prediction_perf = __evaluate_sir_model__(clf, X_train, y_train, X_test, y_test)
        for param, value in prediction_perf.items():
            output[fold_id][param] = value
        print('\tAUC (train):', output[fold_id]['Train_AUC'])
        print('\tAUC (test):', output[fold_id]['Test_AUC'])

        ''' Extract weights and expand feature blocks to original features '''
        df_weights = __extract_weights_from_bagging_ensemble__(clf, lsdf_case.index)
        print('\tSelected blocks:', df_weights.shape)
        df_original_weights = {}; amr_blocks = set()
        for block, weight in df_weights.iteritems():
            block_id = int(block[1:])
            for feature in case_block_defs[block_id]:
                df_original_weights[feature] = weight
                if feature in known_amr_drug_set:
                    amr_blocks.add(block)
        df_original_weights = pd.Series(df_original_weights)
        print('\tSelected features:', df_original_weights.shape)

        ''' Rank features and identify ranks of known AMR genes '''
        ranks_avg = scipy.stats.rankdata(-np.abs(df_original_weights.values), 
                                         method='average') # ranks assigned smaller first, negate
        ranks_dense = scipy.stats.rankdata(-np.abs(df_original_weights.values), 
                                         method='dense') # dense ranking, collapse instead of averaging ties
        df_ranks_avg = pd.Series(data=ranks_avg, index=df_original_weights.index)
        df_ranks_dense = pd.Series(data=ranks_dense, index=df_original_weights.index)
        df_ranks_avg.name = 'ranks_avg'
        df_ranks_dense.name = 'ranks_dense'
        df_ranks = pd.concat([df_ranks_avg, df_ranks_dense], axis=1)
        known_amr_detected = [x for x in df_ranks.index if x in known_amr_drug_set]
        df_ranks_known_amr = df_ranks.reindex(known_amr_detected)
        print('\tSelected known AMR features:', df_ranks_known_amr.shape)
        output[fold_id]['known_AMR_ranks_avg_dense'] = {}
        for feature in df_ranks_known_amr.index:
            feature_rank = df_ranks_known_amr.loc[feature,:].values
            output[fold_id]['known_AMR_ranks_avg_dense'][feature] = list(feature_rank)
        
        ''' Compute similiar ranking at block-level '''
        block_ranks = scipy.stats.rankdata(-np.abs(df_weights.values),
                                           method='average') # ranks assigned smaller first, negate
        df_block_ranks = pd.Series(data=block_ranks, index=df_weights.index)
        selected_amr_blocks = [x for x in df_weights.index if x in amr_blocks]
        df_block_ranks_amr = df_block_ranks.reindex(selected_amr_blocks)
        print('\tSelected known AMR blocks:', df_block_ranks_amr.shape)
        output[fold_id]['known_AMR_blocks'] = df_block_ranks_amr.to_dict()

        ''' Report total run time for fold '''
        runtime = time.time() - start_time
        output[fold_id]['Runtime'] = runtime
        print('\tRuntime:', round(runtime,3))
    return output


def compute_known_amr_distr(case_block_defs, known_amr_drug_set, selected_blocks=[]):
    '''
    Calculates the number of known AMR features and blocks containing
    known AMR features before and (optionally) after LOR-filtering.
    
    Parameters
    ----------
    case_block_defs : list
        List s.t. case_block_defs[i] = features in ith block,
        from prepare_amr_case_data()
    known_amr_drug_set : set
        Known AMR features, from prepare_amr_case_data()
    selected_blocks : list
        Blocks set (i.e. post-LOR-filtering), denoted 'B#' (default [])
        
    Returns
    -------
    amr_counts : tuple
        Four values describing the AMR feature distribution:
            amr_counts[0] = Total known AMR features
            amr_counts[1] = Total blocks containing at least one AMR feature
            amr_counts[2] = Number of known AMR features among selected_blocks
            amr_counts[3] = Number of known blocks among selected_blocks 
                containing at least one AMR feature.
    amr_blocks : dict
        Mapping between block : known AMR features
    '''
    amr_blocks = {}
    for i, block in enumerate(case_block_defs):
        block_amr = [x for x in block if x in known_amr_drug_set]
        if len(block_amr) > 0:
            block_id = 'B' + str(i)
            amr_blocks[block_id] = block_amr
    n_amr_features = len(known_amr_drug_set)
    n_amr_blocks = len(amr_blocks)
    n_sel_amr_features = 0; n_sel_amr_blocks = 0
    for block in selected_blocks:
        if block in amr_blocks:
            n_sel_amr_blocks += 1
            n_sel_amr_features += len(amr_blocks[block])
    amr_counts = (n_amr_features, n_amr_blocks, n_sel_amr_features, n_sel_amr_blocks)
    return amr_counts, amr_blocks
        
    
def __extract_weights_from_bagging_ensemble__(clf, feature_labels):
    '''
    Extracts weights from a bagging ensemble (specifically tested for
    LinearSVC in BaggingEnsemble). For each feature, takes the average of 
    its weight in all sub-classifiers that included the feature.
    '''
    df_weights_full = {}
    for i,sub_clf in enumerate(clf.estimators_):
        selected_features = clf.estimators_features_[i]
        coef = sub_clf.coef_[0]
        df = pd.Series(data=coef, index=selected_features)
        df_weights_full['CLF' + str(i)] = df
    df_weights_full = pd.DataFrame.from_dict(df_weights_full)
    df_weights_full.index = df_weights_full.index.map(lambda x: feature_labels[x])
    df_weights = pd.Series(data=np.nanmean(df_weights_full, axis=1), index=df_weights_full.index)
    df_weights = df_weights[df_weights != 0.0] # reduced to selected features
    return df_weights   

        
def __evaluate_sir_model__(clf, X_train, y_train, X_test, y_test):
    '''
    Computes the accuracy, precision, recall, MCC, and AUC of a trained 
    binary classifier on the test and training tests.
    '''
    metrics = [ (sklearn.metrics.accuracy_score, 'Accuracy'),
                (sklearn.metrics.precision_score, 'Precision'),
                (sklearn.metrics.recall_score, 'Recall'),
                (sklearn.metrics.matthews_corrcoef, 'MCC')]
    y_train_hat = clf.predict(X_train)
    y_test_hat = clf.predict(X_test)
    p_train = clf.predict_proba(X_train)
    p_test = clf.predict_proba(X_test)

    results = {}
    for metric, name in metrics:
        results['Train_'+name] = metric(y_train, y_train_hat)
        results['Test_'+name] = metric(y_test, y_test_hat)
    auc_train = sklearn.metrics.roc_auc_score(y_train, p_train[:,1])
    auc_test = sklearn.metrics.roc_auc_score(y_test, p_test[:,1])
    results['Train_AUC'] = auc_train
    results['Test_AUC'] = auc_test
    return results
    
        
def prefilter_features_by_lor(lsdf_case_block, df_amr_org_drug, min_freq=3, max_features=10000):
    '''
    Filters compressed feature by raw frequancy and LOR. For example, 
    if 10000 features are allowed, selected the 5000 with the lowest 
    and 5000 with the highest LORs.
    
    Parameters
    ----------
    lsdf_case_block : amr_pangenome.sparse_utils.LightSparseDataFrame
        Sparse genome x block binary table from prepare_amr_case_data()
    df_amr_org_drug : pd.Series
        AMR phenotypes by genome from prepare_amr_case_data()
    min_freq : int
        Minimum occurence of a feature to include (default 3)
    max_features : int
        Maximum features returned (default 10000)
    
    Returns 
    -------
    lsdf_case_block2 : amr_pangenome.sparse_utils.LightSparseDataFrame
        Sparse genome x block binary table with filtered features 
    '''
    ''' Filter by frequency '''
    if min_freq > 0:
        feature_freqs = np.array(lsdf_case_block.data.sum(axis=1))[:,0]
        count_filtered = np.where(feature_freqs >= min_freq)[0]
        lsdf = lsdf_case_block.islice(i_indices=count_filtered)
    else:
        lsdf = lsdf_case_block
    
    ''' Filter by LOR '''
    if lsdf.shape[0] <= max_features:
        return lsdf
    contingency = contingency_tables_from_sparse(
        lsdf.data, df_amr_org_drug.values.astype(float), batch_size=10000)
    lors = adjusted_lor(contingency)
    df_lors = pd.Series(index=np.arange(len(lors)), data=lors).sort_values(ascending=False)
    selected_indices = df_lors.index[:max_features/2].tolist() +\
                       df_lors.index[-max_features/2:].tolist()
    lsdf_case_block2 = lsdf.islice(i_indices=selected_indices)
    print('Species x drug LOR-selected compressed features:', lsdf_case_block2.shape)
    return lsdf_case_block2


def prepare_amr_case_data(drug, lsdf_features, df_amr_org, df_known_amr):
    '''
    Prepares data for a specific species x drug case:
    1) Reduces AMR data to specified drug
    2) Reduces known AMR genes to specified drug
    3) Filters genomes to those with AMR data for the drug and removes empty features
    4) Compresses remaining features with identical occurence into feature blocks
    
    Parameters
    ----------
    drug : str
        Name of antimicrobial
    lsdf_features : amr_pangenome.sparse_utils.LightSparseDataFrame
        Feature table, from prepare_species_data()
    df_amr_org : pd.DataFrame
        Genome x AMR phenotype table, from prepare_species_data()
    df_known_amr : pd.DataFrame
        Feature x drug table of known AMR features, from prepare_species_data()
        
    Returns
    -------
    df_amr_org_drug : pd.Series
        AMR phenotypes by genome
    known_amr_drug_set : set
        Known AMR features associated with the drug
    lsdf_case_features : amr_pangenome.sparse_utils.LightSparseDataFrame
        Sparse genome x feature binary table reduced to genomes with 
        AMR data related to the drug of interest
    lsdf_case_block : amr_pangenome.sparse_utils.LightSparseDataFrame
        Sparse genome x block binary table after merging identical features
    case_block_defs : list
        List s.t. case_block_defs[i] = features in ith block
    '''
    
    ''' Reduce AMR data (phenotypes + known genes) to drug of interest '''
    df_amr_org_drug = df_amr_org.loc[:,drug].dropna()
    print('Species x drug AMR data:', df_amr_org_drug.shape)
    df_known_amr_drug = df_known_amr.loc[:,drug].dropna()
    known_amr_drug_set = set(df_known_amr_drug.index)
    print('Species x drug AMR features:', len(known_amr_drug_set))

    ''' Remove features not present in genomes with AMR data for selected drug  '''
    lsdf_case_features = lsdf_features.labelslice(columns=df_amr_org_drug.keys())
    lsdf_case_features = lsdf_case_features.drop_empty(axis='index')
    print('Species x drug reduced features:', lsdf_case_features.shape)

    ''' Compress statistically identical features for remaining features '''
    lsdf_case_block, case_block_defs =\
        amr_pangenome.sparse_utils.compress_rows(lsdf_case_features)
    print('Species x drug compressed features:', lsdf_case_block.shape)
    return df_amr_org_drug, known_amr_drug_set, lsdf_case_features,\
        lsdf_case_block, case_block_defs

    
def prepare_species_data(name_short, df_amr, workdir=PROJECT_DIR):
    ''' 
    Loads the feature table, feature annotations, species-specific
    AMR data, and known AMR features for the specified organism.
    
    Parameters
    ----------
    name_short : str
        Species abbreviation preceding feature names/file paths.
        Requires feature table in LSDF format, feature annotations,
        and known AMR genes to have been generated.
    df_amr : pd.DataFrame
        Full genome x drug AMR phenotype table
    workdir : str
        Path to working directory (default '/media/pekar2/AMR_MLGWAS/')
        
    Returns
    -------
    lsdf_features : amr_pangenome.sparse_utils.LightSparseDataFrame
        Sparse genome x feature binary table
    feature_to_annots : dict
        Mapping of {feature:annotation}, gene level + allele exceptions
    df_amr_org : pd.DataFrame
        Genome x AMR phenotype table, with genomes for the selected species
    df_known_amr : pd.DataFrame
        Feature x drug table, with known AMR features of the selected species
    '''
    ''' Set up paths to requisite files based on name_short and workdir '''
    feature_file = workdir + name_short + '_genomes/' + name_short\
        + '_features/' + name_short + '_strain_by_feature.npz'
    annot_file = workdir + name_short + '_genomes/' + name_short\
        + '_annotations.tsv'
    annot_file2 = workdir + name_short + '_genomes/' + name_short\
        + '_noncoding_annotations.tsv'
    known_amr_file = workdir + name_short + '_genomes/'\
        + name_short + '_features/' + name_short + '_known_amr_features.csv'
                         
    ''' Load full species feature table '''
    lsdf_features = amr_pangenome.sparse_utils.read_lsdf(feature_file)
    print('Species feature table:', lsdf_features.shape)

    ''' Load full species feature annotations'''
    feature_to_annots = {}
    for filename in [annot_file, annot_file2]:
        with open(filename, 'r') as f:
            for line in f:
                data = line.strip().split('\t')
                feature = data[0]; annots = ';'.join(data[1:])
                feature_to_annots[feature] = annots
    print('Species feature annotations:', len(feature_to_annots))

    ''' Reduce AMR data to species '''
    df_amr_org = df_amr.reindex(index=lsdf_features.columns)
    df_amr_org = df_amr_org.dropna(how='all', axis=1).drop(columns=['species'])
    print('Species AMR data:', df_amr_org.shape)
    
    ''' Load known AMR genes for species '''
    df_known_amr = pd.read_csv(known_amr_file, index_col=0)
    print('Species known AMR features:', df_known_amr.shape)
    return lsdf_features, feature_to_annots, df_amr_org, df_known_amr


def contingency_tables_from_sparse(sp_features, target, batch_size=10000):
    '''
    Computes contigency tables (TPs, FPs, FNs, TNs) between
    all features in sparse matrix format and a target vector.
    
    Parameters
    ----------
    sp_features : scipy.sparse.spmatrix
        Binary feature x sample table in sparse format.
    target : np.array
        Binary target vector for samples
    batch_size : int
        Max batch of features to densify at any time (default 10000)

    Returns
    -------
    contingency : np.array (n_features, 4)
        Array with TPs, FPs, FNs, TNs per feature as columns
    '''
    data = sp_features.tocsr()
    n_features, n_samples = data.shape
    positives = float(target.sum())
    negatives = 1.0 - n_samples
    positive_rate = positives / float(n_samples)
    contingency = np.zeros((n_features, 4))
    batch_target = np.tile(target, (batch_size,1))
    for i in np.arange(0,data.shape[0],batch_size):
        batch_features = np.array(data[i:i+batch_size,:].todense())
        if batch_features.shape[0] != batch_size:
            batch_target = np.tile(target, (batch_features.shape[0],1))
        feature_incidence = batch_features.sum(axis=1)
        TPs = np.logical_and(batch_features, batch_target).sum(axis=1)
        FPs = feature_incidence - TPs
        FNs = positives - TPs
        TNs = n_samples - TPs - FPs - FNs
        contingency[i:i+batch_features.shape[0], 0] = TPs
        contingency[i:i+batch_features.shape[0], 1] = FPs
        contingency[i:i+batch_features.shape[0], 2] = FNs
        contingency[i:i+batch_features.shape[0], 3] = TNs
    return contingency


def adjusted_lor(contingency):
    ''' 
    Computes adjusted LORs (log2 odds ratio) from contingency tables 
    https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007608
    '''
    TPs = contingency[:,0]; FPs = contingency[:,1]
    FNs = contingency[:,2]; TNs = contingency[:,3]
    PRs = np.divide(TPs + FNs, contingency.sum(axis=1, dtype='float'))
    NRs = 1.0 - PRs
    numerator = np.multiply(TPs + PRs, TNs + NRs)
    denominator = np.multiply(FPs + NRs, FNs + PRs)
    return np.log2(np.divide(numerator, denominator))