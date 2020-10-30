#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed May 6 16:35:52 2020

@author: jhyun95
"""

import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse, scipy.optimize, scipy.stats
import scipy.cluster.hierarchy as sch

from mlgwas import sparse_arrays_to_sparse_matrix

def test_segment_enrichment(df_annot_counts, min_freq=10, 
        total_gene_counts='TOTAL', lor_only=False, merging={}):
    '''
    Tests whether the core, accessory, or unique genome is enriched for 
    a particular functional annotation (i.e. COG, GO term). 
    
    Parameters
    ----------
    df_annot_counts : pd.DataFrame
        Annotation x segment (core, accessory, unique) count table.
        See count_segment_cog() and coutn_segment_go() output for format.
    min_freq : int
        Excludes annotations with total frequency < min_freq (default 10)
    total_gene_counts : str or tuple
        Total number of genes in the core, accessory, unique gene sets 
        If str, uses the row with this value as the label. If 3-tuple, uses
        counts from in the order of the columns in df_annot_counts (default 'TOTAL')
    lor_only : bool
        If True, computes LORs only. Much faster than computing Fisher's Exact
        test p-values, especially for many annotations as in GO terms (default False)
    merging : dict
        For each key:val, merges counts for key into counts for val. For example,
        {'?':'S'} would merge missing COG annotations into the function unknown
        category S. If val is None, ignores the key entirely (default {})
        
    Returns
    -------
    df_lors : pd.DataFrame
        Log2 odds ratio between each segment and functional annotation.
        Uses weighted pseudocounts as in doi:10.1371/journal.pcbi.1007608
    df_pvals : pd.DataFrame
        Raw p-values from Fisher's exact tests.
    '''
    
    ''' Merge or drop specified annotations '''
    df_annot = df_annot_counts.copy()
    for merge, target in merging.items():
        if target != None: # merge into target category
            df_annot.loc[target,:] += df_annot.loc[merge,:]
        df_annot = df_annot.drop(merge, axis=0)
    
    ''' Filter out rare mutations based on min_freq '''
    annot_counts = df_annot.sum(axis=1)
    df_annot = df_annot[annot_counts >= min_freq]
    
    ''' Get total core, accessory, unique, and overall genes '''
    if type(total_gene_counts) == str:
        segment_gene_counts = df_annot.loc[total_gene_counts,:].values
        df_annot = df_annot.drop(index=total_gene_counts)
    else:
        segment_gene_counts = total_gene_counts
    total_genes = np.sum(segment_gene_counts)
    
    ''' Compute LOR and apply association tests per annotation '''
    lors = np.zeros(df_annot.shape)
    pvals = np.zeros(df_annot.shape)
    
    for s,segment in enumerate(df_annot.columns):
        ''' Construct contingency tables '''
        segment_gene_count = segment_gene_counts[s]
        tps = df_annot.loc[:,segment].values #  TPs: in annot, in segment
        fps = df_annot.sum(axis=1) - tps # FPs: in annot, out segment
        fns = segment_gene_count - tps # FNs: out annot, in segment
        tns = total_genes - segment_gene_count - fps # TNs: out annot, out segment
        
        ''' Compute LORs and apply Fisher's exact tests '''
        lors[:,s] = adjusted_lor(tps, fps, fns, tns)
        if not lor_only: # TODO: Faster/vectorized Fisher's exact tests?
            for a in np.arange(len(tps)):
                oddsratio, pvalue = scipy.stats.fisher_exact([[tps[a],fps[a]],[fns[a],tns[a]]])
                pvals[a,s] = pvalue

    ''' Format output '''
    df_lors = pd.DataFrame(index=df_annot.index, columns=df_annot.columns, data=lors)
    if lor_only:
        return df_lors
    else:
        df_pvals = pd.DataFrame(index=df_annot.index, columns=df_annot.columns, data=pvals)
        return df_lors, df_pvals
    

def count_segment_cog(df_genes, df_eggnog, core_min, unique_max, include_totals=True):
    ''' 
    Count the number of core, accessory, and unique genes per COG.
    Genes with multiple COG annotations are counted into all related
    COG categories. Genes that were not annotated are assigned COG "?"
    
    Parameters
    ----------
    df_genes : pd.DataFrame
        Binary gene x strain table
    df_eggnog : pd.DataFrame
        DataFrame loaded from *.emapper.annotation file. Index column
        should be processed into format matching index of df_genes.
    core_min : float
        Defines core genes as frequency > core_min
    unique_max : float 
        Defines unique genes as frequency < unique_max. Accessory
        genes are genes with unique_max <= frequency <= core_min
    include_totals : bool
        If True, adds a row with the total genes per segment (default True)
        
    Returns
    -------
    df_cog_segments : pd.DataFrame
        COG x pangenome segment (core, accessory, unique) count table.
    '''
    
    ''' Identify core, accessory, and unique genes '''
    df_gene_counts = df_genes.fillna(0).sum(axis=1)
    core_genes = df_gene_counts[df_gene_counts > core_min].index
    unique_genes = df_gene_counts[df_gene_counts < unique_max].index
    accessory_genes = df_gene_counts[np.logical_and(df_gene_counts <= core_min, df_gene_counts >= unique_max)]
    
    ''' Identify represented cogs (unpack multi-cog annotations) '''
    df_cogs = df_eggnog['COG cat']
    cogs = df_cogs.unique()
    cogs = sorted(filter(lambda x: not ',' in x, cogs))
    
    ''' Count occurence of each COG + unannotated "?" '''
    cog_counts = np.zeros((len(cogs) + 1, 3))
    segments = ['core', 'accessory', 'unique']
    for g, gene_set in enumerate([core_genes, accessory_genes, unique_genes]):
        df_set_cogs = df_cogs[gene_set].fillna('?')
        for c, cog in enumerate(cogs+['?']):
            df_cog_specific = df_set_cogs[df_set_cogs.map(lambda x: cog in x)]
            cog_counts[c,g] = df_cog_specific.shape[0]
    df_cog_segments = pd.DataFrame(index=cogs+['?'], columns=segments, data=cog_counts)
    
    ''' Optionally add the total genes per segment '''
    if include_totals:
        counts = (len(core_genes), len(accessory_genes), len(unique_genes))
        df = pd.DataFrame(index=segments, columns=['TOTAL'], data=counts).T
        df_cog_segments = pd.concat([df_cog_segments,df], axis=0)
    
    return df_cog_segments


def count_segment_go(df_genes, df_eggnog, core_min, unique_max, include_totals=True):
    ''' 
    Count the number of core, accessory, and unique genes per GO term.
    Genes with multiple GO annotations are counted into all related
    GO terms. Genes that were not annotated are assigned GO term "GO:?"
    
    Parameters
    ----------
    df_genes : pd.DataFrame
        Binary gene x strain table
    df_eggnog : pd.DataFrame
        DataFrame loaded from *.emapper.annotation file. Index column
        should be processed into format matching index of df_genes.
    core_min : float
        Defines core genes as frequency > core_min
    unique_max : float 
        Defines unique genes as frequency < unique_max. Accessory
        genes are genes with unique_max <= frequency <= core_min
    include_totals : bool
        If True, adds a row with the total genes per segment (default True)
        
    Returns
    -------
    df_go_segments : pd.DataFrame
        GO term x pangenome segment (core, accessory, unique) count table.
    '''
    
    ''' Identify core, accessory, and unique genes '''
    df_gene_counts = df_genes.fillna(0).sum(axis=1)
    core_genes = df_gene_counts[df_gene_counts > core_min].index
    unique_genes = df_gene_counts[df_gene_counts < unique_max].index
    accessory_genes = df_gene_counts[np.logical_and(df_gene_counts <= core_min, df_gene_counts >= unique_max)]
    
    ''' Identifying GO terms '''
    df_go = df_eggnog['GO_terms']
    segments = ['core','accessory','unique']
    go_counts = {}
    for g, gene_set in enumerate([core_genes, accessory_genes, unique_genes]):
        label = segments[g]
        go_counts[label] = {}
        df_set_go = df_go[gene_set].fillna('GO:?')
        for gene, go_list in df_set_go.iteritems():
            for go_term in go_list.split(','):
                if not go_term in go_counts[label]:
                    go_counts[label][go_term] = 0
                go_counts[label][go_term] += 1
    df_go_segments = pd.DataFrame.from_dict(go_counts).fillna(0).reindex(columns=segments)
    
    ''' Optionally add the total genes per segment '''
    if include_totals:
        counts = (len(core_genes), len(accessory_genes), len(unique_genes))
        df = pd.DataFrame(index=segments, columns=['TOTAL'], data=counts).T
        df_go_segments = pd.concat([df_go_segments,df], axis=0)
        
    return df_go_segments


def get_allele_count(features, parent_abb='C', allele_abb='A', verify_features=False):
    '''
    Computes the number of alleles per parent feature (i.e. number of
    unique alleles per CDS gene cluster).
    
    Parameters
    ----------
    features : iterable
        List of feature names at the allele level
    parent_abb : str
        Abbreviation of parent feature. 'C' for CDS/genes, or
        'T' for transcripts/non-coding features (default 'C').
    allele_abb : str
        Abbreviation of allele feature. 'A' for alleles, 'U' for 
        upstream variants, or 'D' for downstream variants (default 'A')
    verify_features : bool
        If True, checks that features are of the type specified and filters 
        out irrelevant features. Otherwise, assumes all features are the
        correct type (default False)
        
    Returns
    -------
    allele_counts : dict
        Dictionary mapping parent features to allele counts
    '''
    if verify_features:
        feature_footers = map(lambda x: x.split('_')[-1], features)
        relevant_features = filter(lambda x: x[0] == parent_abb and allele_abb in x, feature_footers)
    else:
        relevant_features = features
    parent_features = map(lambda x: x[:x.rindex(allele_abb)], relevant_features)
    return dict(collections.Counter(parent_features))


def find_pangenome_segments(df_genes, threshold=0.1, ax=None):
    '''
    Computes the gene frequency thresholds at which a gene can be categorized as 
    core, accessory, or unique. Specifically, models the gene frequency distribution
    as the sum of two power laws (one flipped), and fits the CDF to a five-parameter
    function dervied from those power laws. Also identifies the inflection point and
    the core and unique extremes relative to the inflection point and threshold.
    
          PMF(x;c1,c2,a1,a2) ~ c1 * x^-a1 + c2 * (n-x)^-a2
        CDF(x;c1,c2,a1,a2,k) ~ c1/(1-a1) * x^(1-a1) - c2/(1-a2) * (n-x)^(1-a2) + k
        
    Where x = frequency, n = maximum frequency + 1, other variables are parameters.
    
    Pangenome segments example at 10%:
    - N = total strains, R = computed inflection point
    - Core: Observed in >= R + (1 - 0.1) * (N-R) strains
    - Unique: Observed in <= 0.1 * R strains
    - Accessory: Everything in between
    
    Parameters
    ----------
    df_genes : pd.DataFrame
        Binary gene x strain table.
    threshold : float
        Proximity to each frequency extreme compared to inflection point
        that determines if a gene is core, unique, or accessory (default 0.1)
    ax : plt.axes
        If provided, plots pangenome frequency CDF with segments (default None)
        
    Returns
    -------
    segments : tuple
        2-tuple with (min core limit, max unique limit), not rounded.
    popt : tuple
        5-tuple with fitted CDF parameters (c1,c2,a1,a2,k). Note that
        c1, c2, and k are scaled relative to the number of unique genes
    r_squared : float
        R^2 between fit and observed cumulative gene frequency distribution
    ax : plt.axes
        If ax is not None, returns axis with plots
    '''

    ''' Computing gene frequencies and frequency counts '''
    df_gene_freq = df_genes.fillna(0).sum(axis=1)
    df_freq_counts = df_gene_freq.value_counts()
    df_freq_counts = df_freq_counts[sorted(df_freq_counts.index)]
    cumulative_frequencies = np.cumsum(df_freq_counts.values)
    frequency_bins = np.array(df_freq_counts.index)
    
    ''' Fitting CDF '''
    X = frequency_bins.astype(float)
    Y = cumulative_frequencies.astype(float)
    n = max(frequency_bins) + 1
    dual_power_cdf = lambda x,c1,c2,a1,a2,k: \
        Y[0]*(c1*np.power(x,1.0-a1)/(1.0-a1) - c2*np.power(n-x,1.0-a2)/(1.0-a2) + k)
    p0 = [1.0,1.0,2.0,2.0,1.0]
    bounds = ([0.0,0.0,1.0,1.0,0.0],[np.inf,np.inf,np.inf,np.inf,Y[-1]/Y[0]])
    popt, pcov = scipy.optimize.curve_fit(dual_power_cdf, X, Y, p0=p0, bounds=bounds, maxfev=100000)
    
    ''' Extracting inflection point of CDF and frequency thresholds '''
    dual_power_pdf = lambda x,c1,c2,a1,a2: Y[0]*(c1*np.power(x,-a1) + c2*np.power(n-x,-a2))
    dual_power_pdf_fit = lambda x: dual_power_pdf(x,*popt[:4]) # minimize PMF
    res = scipy.optimize.minimize_scalar(dual_power_pdf_fit, bounds=[1,n-1])
    inflection_freq = res.x # inflection point x, i.e. frequency threshold 
    unique_strains_max = inflection_freq * threshold
    core_strains_min = inflection_freq + (n - 1 - inflection_freq) * (1.0 - threshold)
    segments = (core_strains_min, unique_strains_max)
    
    ''' Curve fit evaluation: R^2 
        Yes I know R^2 isn't a good curve fit metric. Residual distributions are 
        not random so this function is not a "true" model for gene frequency but
        nonetheless sufficient for purposes of segmenting into core/accessory/unique. '''
    Yfit = np.array(map(lambda x: dual_power_cdf(x,*popt), X)) # fitted CDF
    SStot = np.sum(np.square(Y - Y.mean()))
    SSres = np.sum(np.square(Y - Yfit))
    r_squared = 1 - (SSres/SStot)
    
    ''' Optionally, generating plot '''
    if ax:
        ax.plot(X, Y, label='observed')
        ax.plot(X, Yfit, label='fit', ls='--')
        ax.scatter([inflection_freq], [dual_power_cdf(inflection_freq,*popt)], 
                   label='inflection point', color='black', alpha=0.7)
        ax.axvline(unique_strains_max, ls='--', color='k')
        ax.axvline(core_strains_min, ls='--', color='k')
        ax.axvline(inflection_freq, ls='--', color='lightgray')
        
        unique_rounded = int(unique_strains_max) + 1
        core_rounded = int(core_strains_min)
        unique_text = 'Unique:\n<' + str(unique_rounded)
        core_text = 'Core:\n>' + str(core_rounded)
        r2_text = 'R^2=' + str(np.round(r_squared,3))
        ax.text(unique_strains_max + n*0.02, Y[0], unique_text, ha='left', va='bottom')
        ax.text(core_strains_min - n*0.02, Y[0], core_text, ha='right', va='bottom')
        ax.text(unique_strains_max + n*0.1, Y[-1], r2_text, ha='left', va='top')
        ax.set_xlabel('Gene frequency')
        ax.set_ylabel('Cumulative genes')
        return segments, popt, r_squared, ax
    else:
        return segments, popt, r_squared


def find_pangenome_segments_logistic(df_genes, threshold=0.1, ax=None):
    '''
    Computes the gene frequency thresholds at which a gene can be categorized as 
    core, accessory, or unique. Specifically, fits a three-parameter logistic 
    function for cumulative genes vs. max strains with gene (gene frequency),
    identifies the inflection point, and identifies the core and unique extremes
    relative to the inflection point and threshold. See link for details,
    where parameters (B,v,M) are fit.
    
    https://en.wikipedia.org/wiki/Generalised_logistic_function
    
    Example at 10%:
    - N = total strains, R = computed inflection point
    - Core: Observed in >= R + (1 - 0.1) * (N-R) strains
    - Unique: Observed in <= 0.1 * R strains
    - Accessory: Everything in between
    
    Parameters
    ----------
    df_genes : pd.DataFrame
        Binary gene x strain table.
    threshold : float
        Proximity to each frequency extreme compared to inflection point
        that determines if a gene is core, unique, or accessory (default 0.1)
    ax : plt.axes
        If provided, pangenome frequency plot with segments (default None)
        
    Returns
    -------
    segments : tuple
        2-tuple with (min core limit, max unique limit), not rounded.
    popt : tuple
        3-tuple with fitted logistic parameters. These parameters are 
        fit after scaling the x and y to the range 0-1.
    ax : plt.axes
        If ax is not None, returns axis with plots.
    '''
    
    ''' Computing gene frequencies and frequency counts '''
    df_gene_freq = df_genes.fillna(0).sum(axis=1)
    df_freq_counts = df_gene_freq.value_counts()
    df_freq_counts = df_freq_counts[sorted(df_freq_counts.index)]
    cumulative_frequencies = np.cumsum(df_freq_counts.values)
    frequency_bins = df_freq_counts.index

    ''' Fitting 3-parameter logistic '''
    def generalized_logistic(t,B,v,M):
    #     return A + (K-A) * np.power(1.0+np.exp(-B*(t-M)), -1.0/v) # 5-parameter form
        return np.power(1.0+np.exp(-B*(t-M)), -1.0/v) # 3-parameter form

    X = cumulative_frequencies.astype(float)
    Y = frequency_bins.values.astype(float)
    Xscaled = (X - np.min(X)) / (np.max(X) - np.min(X))
    Yscaled = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
    p0 = [1.0, 1.0, 1.0]
    popt, pcov = scipy.optimize.curve_fit(generalized_logistic, Xscaled, Yscaled, p0)
    B,v,M = popt
    Yfit = np.array(map(lambda x: generalized_logistic(x, *popt), Xscaled))
    Yfit = Yfit * (np.max(Y) - np.min(Y)) + np.min(Y)
    
    ''' Segmenting into core/accessory/unique '''
    inflection_scaled_x = M - np.log(v) / B
    inflection_scaled_y = generalized_logistic(inflection_scaled_x, *popt)
    inflection_genes = inflection_scaled_x * (np.max(X) - np.min(X)) + np.min(X)
    inflection_strains = inflection_scaled_y * (np.max(Y) - np.min(Y)) + np.min(Y)
    unique_strains_max = inflection_strains * threshold
    core_strains_min = inflection_strains + (Y[-1] - inflection_strains) * (1.0 - threshold)
    segments = (core_strains_min, unique_strains_max)

    ''' Optionally, generating plot '''
    if ax:
        yscale = np.max(Y)
        ax.plot(cumulative_frequencies, frequency_bins, label='observed')
        ax.plot(X, Yfit, label='fit', ls='--')
        ax.axhline(unique_strains_max, ls='--', color='k')
        ax.axhline(core_strains_min, ls='--', color='k')
        unique_rounded = int(unique_strains_max) + 1
        core_rounded = int(core_strains_min)
        unique_text = 'U: <' + str(unique_rounded)
        core_text = 'C: >' + str(core_rounded)
        accessory_text = 'A: '  + str(unique_rounded) + '-' + str(core_rounded)
        ax.text(X[0], unique_strains_max + yscale*0.02, unique_text, ha='left', va='bottom')
        ax.text(X[0], core_strains_min - yscale*0.02, core_text, ha='left', va='top')
        ax.text(X[0], (unique_strains_max + core_strains_min) * 0.5, accessory_text, ha='left', va='center')
        ax.set_xlabel('Cumulative genes')
        ax.set_ylabel('Max genomes with gene')
        return segments, popt, ax
    else:
        return segments, popt
    

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
    included_strain_counts = range(drop_early, num_strains)
    included_strain_counts = filter(lambda x: not new_gene_rates[x] == 0, included_strain_counts)
    included_gene_rates = new_gene_rates[included_strain_counts]

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


def count_variant_abundance(df_genes, df_variants, keep_variant_names=False):
    '''
    Counts the distribution of variants per gene. Assumes that genes
    are named <org>_C#, and variants <org>_C#X#, where X = A,U,D.
    
    Parameters
    ----------
    df_genes : pd.DataFrame
        Binary gene x strain table
    df_variants: pd.DataFrame
        Binary variant x strain table
    keep_variant_names : bool
        If true, stores variant names in output. Otherwise,
        assumes that variants are named C#X0, C#X1, ... and
        stores counts indexed by variant name 0,1, ... 
        
    Returns
    -------
    variant_abundance : dict
        Maps {gene:[v0 count, v1 count, etc.]} or
        Maps {gene:variant:count}
    '''
    gene_counts = df_genes.fillna(0).sum(axis=1)
    variant_counts = df_variants.fillna(0).sum(axis=1)
    
    ''' Identify the type of variant '''
    gene, variant = variant_counts.index[0].split('_')
    variant_type = None
    for vtype in ['A','U','D']:
        if vtype in variant:
            variant_type = vtype; break
    if variant_type is None:
        print 'Variant type could not be identified, aborting.'; return
    
    ''' Count variant distributions '''
    variant_abundance = {}
    for variant, count in variant_counts.iteritems():
        gene = variant[:variant.rindex(variant_type)]
        if not gene in variant_abundance:
            variant_abundance[gene] = {}
        variant_abundance[gene][variant] = count
    
    ''' Optionally, compress abundance information '''
    if not keep_variant_names:
        for gene in variant_abundance:
            abundances = variant_abundance[gene]
            ind_to_abundance = {int(variant[variant.rindex(variant_type)+1:]):count
                                for variant,count in abundances.items()}
            max_ind = max(ind_to_abundance.keys())
            compressed_abundances = [0] * (max_ind + 1)
            for ind, count in ind_to_abundance.items():
                compressed_abundances[ind] = count
            variant_abundance[gene] = compressed_abundances
    return variant_abundance


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


def adjusted_lor(tp, fp, fn, tn):
    ''' 
    Log2 odds ratio with weighted pseudocounts, as described
    in https://doi.org/10.1371/journal.pcbi.1007608.
    Supports both singular values and array inputs.
    '''
    p = tp + fn
    n = fp + tn
    pr = np.true_divide(p, p+n)
    nr = 1.0 - pr
    numer = np.multiply(tp + pr, tn + nr)
    denom = np.multiply(fp + nr, fn + pr)
    return np.log2(np.true_divide(numer, denom))


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