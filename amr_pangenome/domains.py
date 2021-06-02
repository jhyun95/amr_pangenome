#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Thu Wed 26 23:34:17 2021

@author: jhyun95

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from pangenome import load_sequences_from_fasta

IPR_COLUMNS = ['acc', 'md5', 'length', 'analysis', 'sig_acc', 'sig_desc', 'start', 'stop', 
               'score', 'status', 'date', 'ipr_acc', 'ipr_desc', 'go_annot', 'pathways']

def load_ipr_domains(ipr_file, max_dist=10, max_domain_length_ratio=0.8):
    '''
    Loads all domains in an InterProScan TSV formatted file, and 
    merges/filters them with the following steps:
    1) Merge domains with the same IPR ID that overlap or nearly overlap (max_dist)
    2) Filter out domains spanning multiple disparate segments
    3) Filter out domains spanning too much of the full protein (max_domain_length_ratio)
    
    Parameters
    ----------
    ipr_file : str of pd.DataFrame
        Path to IPR output file
    max_dist : int
        Minimum separation in AAs between domain segments to merge (default 10)
    max_domain_length_ratio : float
        Maximum length of domain relative to protein to include (default 0.8)
        
    Returns
    -------
    df_ipr_merged : pd.DataFrame
        DataFrame containing processed domains
    '''
    with open(ipr_file, 'r') as f:
        data = []
        for line in f:
            line_items = line.strip().split('\t')
            line_items += ['-'] * (len(IPR_COLUMNS) - len(line_items))
            data.append(line_items)    
    df_ipr_full = pd.DataFrame(data=data, columns=IPR_COLUMNS)
    for col in ['length', 'start', 'stop']:
        df_ipr_full[col] = df_ipr_full[col].map(int)
    df_ipr_full = df_ipr_full.sort_values(by='acc') # sort by sequence ID
        
    ''' Divide file into separate sequences if data for multiple 
        sequences are present '''
    seq_ids = df_ipr_full.acc.unique()
    if len(seq_ids) == 1: # single sequence present
        return load_ipr_domains_single(
            df_ipr_full, max_dist, max_domain_length_ratio)
    else: # multiple sequences in single file
        df_ipr_processed = []
        last_acc = df_ipr_full.acc.iloc[0]; last_start = 0
        ''' Process annotations one sequence at a time '''
        for current_row in range(df_ipr_full.shape[0]):
            next_acc = df_ipr_full.acc.iloc[current_row]
            if last_acc != next_acc:
                df_ipr = df_ipr_full.iloc[last_start:current_row]
                df_ipr_proc = load_ipr_domains_single(
                    df_ipr, max_dist, max_domain_length_ratio)
                # print current_row, last_acc, df_ipr.shape, df_ipr_proc.shape
                df_ipr_processed.append(df_ipr_proc)
                last_acc = next_acc
                last_start = current_row
                
        ''' Process last batch of annotations '''
        df_ipr = df_ipr_full.iloc[last_start:current_row]
        df_ipr_proc = load_ipr_domains_single(
            df_ipr, max_dist, max_domain_length_ratio)
        df_ipr_processed.append(df_ipr_proc)
        return pd.concat(df_ipr_processed, axis=0)
    

def load_ipr_domains_single(df_ipr, max_dist=10, max_domain_length_ratio=0.8):
    ''' 
    Loads all domains in an InterProScan TSV formatted file and applies 
    several merging/filtering steps. See load_ipr_domains() for details.
    
    Parameters
    ----------
    df_ipr : pd.DataFrame
        DataFrame in format of IPR TSV output, for a single sequence
    max_dist : int
        Minimum separation in AAs between domain segments to merge (default 10)
    max_domain_length_ratio : float
        Maximum length of domain relative to protein to include (default 0.8)
        
    Returns
    -------
    df_ipr_merged : pd.DataFrame
        DataFrame containing processed domains
    '''

    ''' Step 1: Merging overlapping annotations with same IPR ID '''
    acc = df_ipr.iloc[0,0] # overall sequence/protein ID
    protein_length = df_ipr.iloc[0,2] # full length of protein being annotated 
    ipr_domains = {} # maps IPR ID:list of (start, stop) for all associated annotations
    ipr_domain_annots = {} # maps IPR ID:list of all associated text annotations
    for row in df_ipr.itertuples(name=None):
        ipr_acc, ipr_desc = row[12:14]
        ''' Group by IPR ID '''
        if ipr_acc.startswith('IPR'):
            start, stop = row[7:9]
            if not ipr_acc in ipr_domains:
                ipr_domains[ipr_acc] = []
                ipr_domain_annots[ipr_acc] = set()
            ipr_domains[ipr_acc].append((start, stop))    
            ipr_domain_annots[ipr_acc].add(ipr_desc)
        ''' Merge indepentent segments based on overlap/near overlap '''
        for ipr_acc in ipr_domains:
            ipr_domains[ipr_acc] = __merge_domain_segments__(
                ipr_domains[ipr_acc], extend=max_dist)
                
    ''' Step 2: Filter out domains with multiple disparate segments '''
    merge_columns = ['acc', 'ipr_acc', 'protein_length', 'start', 'stop', 'desc']
    data = []
    for ipr_acc in ipr_domains:
        segments = ipr_domains[ipr_acc]
        if len(segments) == 1:
            start, stop = segments[0]
            data.append((acc, ipr_acc, protein_length, start, stop, 
                         ';'.join(ipr_domain_annots[ipr_acc])))
    df_ipr_merged = pd.DataFrame(data=data, columns=merge_columns)

    ''' Step 3: Filter out domains that are >80% total protein length '''
    df_ipr_merged = df_ipr_merged.assign(domain_length=df_ipr_merged.stop - df_ipr_merged.start)
    df_ipr_merged = df_ipr_merged.assign(domain_ratio=df_ipr_merged.domain_length/df_ipr_merged.protein_length)
    df_ipr_merged = df_ipr_merged[df_ipr_merged.domain_ratio <= max_domain_length_ratio]
    return df_ipr_merged

def adjust_domain_positions_for_indels(df_domains, consensus_faa):
    '''
    Compute adjusted start and stop positions of domains to take
    into account indels in the MSA relative to the consensus sequence
    
    Parameters
    ----------
    df_domains : pd.DataFrame
        DataFrame with domain position information as 'start' and 'stop'
        Assumes entries are sorted by sequence accessions 'acc'
    consensus_faa : str
        Path to file with all consensus sequences, including indel dashes
        
    Returns
    -------
    df_domains_adj : pd.DataFrame
        DataFrame with two new columns 'adj_start', 'adj_stop' describing
        adjusted positions of domains within the MSA
    '''
    consensus_sequences = load_sequences_from_fasta(consensus_faa)
    df_domains_adj = []
    
    ''' Process annotations one sequence at a time '''
    last_acc = df_domains.acc.iloc[0]; last_start = 0
    for current_row in range(df_domains.shape[0]):
        next_acc = df_domains.acc.iloc[current_row]
        if last_acc != next_acc:
            df_acc = df_domains.iloc[last_start:current_row]
            consensus = consensus_sequences[last_acc]
            df_acc_adj = adjust_domain_positions_for_indels_single(df_acc, consensus)
            df_domains_adj.append(df_acc_adj)
            last_acc = next_acc
            last_start = current_row 
    
    ''' Process final sequence '''
    df_acc = df_domains.iloc[last_start:current_row]
    consensus = consensus_sequences[last_acc]
    df_acc_adj = adjust_domain_positions_for_indels_single(df_acc, consensus)
    df_domains_adj.append(df_acc_adj)
    df_domains_adj = pd.concat(df_domains_adj, axis=0)
    return df_domains_adj


def adjust_domain_positions_for_indels_single(df_domains, consensus_with_dashes):
    ''' 
    Compute adjusted start and stop positions of domains to take
    into account indels in the MSA relative to the consensus sequence
    
    Parameters
    ----------
    df_domains : pd.DataFrame
        DataFrame with domain position information as 'start' and 'stop'
    consensus_with_dashes : str
        Consensus sequence from MSA, including indel dashes
        
    Returns
    -------
    df_domains_adj : pd.DataFrame
        DataFrame with two new columns 'adj_start', 'adj_stop' describing
        adjusted positions of domains within the MSA
    '''
    consensus_to_msa = []; indel_counter = 0
    ''' Compute a shift for each position, based on cumulative dashes '''
    for i, aa in enumerate(consensus_with_dashes):
        indel_counter += aa == '-'
        if aa != '-':
            consensus_to_msa.append(indel_counter)
    
    ''' Generate adjusted positions from cumulative dashes '''
    df_domains_adj = df_domains.assign(
        adj_start=df_domains.start.map(lambda x: x + consensus_to_msa[x-1]),
        adj_stop=df_domains.stop.map(lambda x: x + consensus_to_msa[x-1])) # 1 -> 0 indexed
    return df_domains_adj


def compute_entropy_across_msa(alignment, df_allele_counts):
    '''
    Computes the entropy at each position of an MSA, weighted 
    by allele frequency. TODO: Copied from notebook, needs testing
    
    Parameters
    ----------
    alignment : dict
        Dictionary mapping allele names to sequences within the MSA
    df_allele_counts : pd.Series or dict
        Series or dictionary mapping allele names to frequency
        
    Returns
    -------
    positional_entropies : list
        Allele-weighted entropy at each position of the MSA
    '''
    
    ''' Weight amino acid at each position by allele frequency '''
    amino_acids = '-ARNDCEQGHILKMFPSTWYV'
    weighted_msa = []
    for allele in alignments[label][gene]:
        allele_count = df_allele_counts[allele] # frequency of allele
        allele_msa = alignment[allele] # sequence of allele within MSA
        weighted_msa += [list(allele_msa)] * int(allele_count)
    weighted_msa = np.array(weighted_msa)
    
    ''' Computed allele-frequency-weighted entropy along MSA '''
    aa_prob_by_pos = np.empty((len(amino_acids), weighted_msa.shape[1]))
    for a, aa in enumerate(amino_acids):
        aa_prob_by_pos[a,:] = (weighted_msa == aa).sum(axis=0) / float(weighted_msa.shape[0])
    neg_log_probs = -np.log(aa_prob_by_pos + (aa_prob_by_pos == 0)) # masking 0s into 1s -> 0 after log
    plogp_vals = np.multiply(neg_log_probs, aa_prob_by_pos)
    positional_entropies = plogp_vals.sum(axis=0)
    return list(positional_entropies)


def compute_domain_entropy_percentiles(positional_entropies, df_all_domains):
    '''
    Computes level of sequence diversity in a domain relative to the rest
    of the protein. Specifically, computes the average entropy along the MSA
    in a domain, as computes its percentile against the average entropy of
    all subsequences of the same length in the protein.
    
    Parameters
    ----------
    positional_entropies : dict
        Mapping between gene clusters and MSA entropy values. Keys in this
        dictionary are used to determine organisms and should map to the
        "gene_cluster" column in df_all_domains.
    df_all_domains : pd.DataFrame
        DataFrame with domain information. Required columns are:
            ipr_acc      - InterPro accession ID
            adj_start    - Adjusted domain start position* 
            adj_stop     - Adjusted domain stop position*
            gene_cluster - Name of gene with domain
            *see adjust_domain_positions_for_indels()
            
    Returns
    -------
    df_domains_ext : pd.DataFrame
        Copy of df_all_domains limited to relevant gene clusters, with
        an additional column "entropy_percentile" described above.
    df_domain_percentiles : pd.DataFrame
        A (domain x gene cluster) table with entropy percentiles.
        Contains NaN for gene clusters missing a given domain. 
    '''
    gene_clusters = sorted(positional_entropies.keys())
    df_domains = df_all_domains[df_all_domains.gene_cluster.map(lambda x: x in gene_clusters)]
    domains = df_domains.ipr_acc.unique()
    domain_by_gene_percentiles = {ipr_acc:{} for ipr_acc in domains}
    
    selected_cols = ['ipr_acc','adj_start','adj_stop','gene_cluster']
    domain_percentiles = []; domain_by_gene = {}
    for row in df_domains.loc[:,selected_cols].itertuples():
        i, ipr_acc, adj_start, adj_stop, gene_cluster = row
        entropy = positional_entropies[gene_cluster]
        
        ''' Compute average entropy of domain '''
        domain_length = adj_stop - adj_start + 1 # inclusive bounds, +1
        domain_entropy = entropy[adj_start-1:adj_stop] # entropy is 0-indexed
        mean_domain_entropy = np.mean(domain_entropy)
        
        ''' Compute percentile of domain average entropy against all
            subsequence of the same length in the protein '''
        segment_mean_entropies = pd.Series(entropy).rolling(window=int(domain_length)).mean()
        segment_mean_entropies = segment_mean_entropies.dropna().values
        n_segments = len(segment_mean_entropies)
        # quantile = (segment_mean_entropies < mean_domain_entropy).sum() / float(n_segments)
        percentile = scipy.stats.percentileofscore(segment_mean_entropies, mean_domain_entropy)
        domain_percentiles.append(percentile)
        domain_by_gene_percentiles[ipr_acc][gene_cluster] = percentile

    ''' Return percentiles and collapsed domain x gene x percentile table '''
    df_domains_ext = df_domains.assign(entropy_percentile=domain_percentiles)
    df_domain_percentiles = pd.DataFrame.from_dict(domain_by_gene_percentiles, orient='index')
    return df_domains_ext, df_domain_percentiles


def plot_entropy_vs_domain(positional_entropies, df_all_domains, 
                           gene_order=None, figsize=(7.5,4)):
    '''
    Creates a two figure plot, comparing entropy along the MSA against domain 
    annotations for the same gene across multiple organisms/gene clusters.
    
    Parameters
    ----------
    positional_entropies : dict
        Mapping between gene clusters and MSA entropy values. Keys in this
        dictionary are used to determine organisms and should map to the
        "gene_cluster" column in df_all_domains.
    df_all_domains : pd.DataFrame
        DataFrame with domain information. Required columns are:
            ipr_acc      - InterPro accession ID
            adj_start    - Adjusted domain start position* 
            adj_stop     - Adjusted domain stop position*
            gene_cluster - Name of gene with domain
            *see adjust_domain_positions_for_indels()
    gene_order : list
        Order to plot/color genes. If None, sorts alphabetically (default None)
    fig_size : tuple
        Figure dimensions in inches (default (7.5,4))
        
    Returns
    -------
    fig : plt.Figure
        Matplotlib pyplot Figure with plots
    axs : tuple of plt.axes.Axes
        Pair of subplots with MSA and domain plots
    '''
    fig, axs = plt.subplots(2,1,figsize=figsize,sharex=True)
    gene_clusters = sorted(positional_entropies.keys()) if gene_order is None else gene_order
    df_domains = df_all_domains[df_all_domains.gene_cluster.map(lambda x: x in gene_clusters)]
    domains = df_domains.ipr_acc.unique()
    
    ''' Build a unified domain order for all organisms, based on typical domain position '''
    domain_rel_pos = {}
    for row in df_domains.loc[:,['ipr_acc', 'adj_start', 'gene_cluster']].itertuples():
        i, ipr_acc, adj_start, gene_cluster = row
        l_seq = len(positional_entropies[gene_cluster])
        relative_position = adj_start / float(l_seq)
        if not ipr_acc in domain_rel_pos:
            domain_rel_pos[ipr_acc] = relative_position 
        domain_rel_pos[ipr_acc] = min(domain_rel_pos[ipr_acc], relative_position)
    domain_order = sorted(domains, key=lambda x: -domain_rel_pos[x])
        
    ''' Plot entropy and domain positions '''
    n_orgs = len(gene_clusters)
    yticks = np.arange(len(domain_order))
    for i,gene_cluster in enumerate(gene_clusters):
        org = gene_cluster.split('_')[0]
        entropy = positional_entropies[gene_cluster]
        axs[0].plot(np.arange(1,len(entropy)+1), entropy, alpha=0.3) # shift to 1-indexed
        
        df = df_domains[df_domains.gene_cluster == gene_cluster]
        df = df.loc[:,['ipr_acc','adj_stop', 'adj_start']].set_index('ipr_acc')
        df = df.reindex(domain_order)
        org_yticks = df.index.map(lambda x: domain_order.index(x))
        axs[1].barh(
            width=df.adj_stop - df.adj_start + 1, # inclusive bounds, +1 to length
            left=df.adj_start, 
            y=org_yticks-0.4+0.8*((i+0.5)/n_orgs),
            height=2*0.8/n_orgs, alpha=0.5, label=org)

    axs[1].set_yticks(yticks)
    axs[1].set_yticklabels(domain_order)
    axs[0].set_ylabel('MSA entropy')
    axs[1].set_ylabel('domain')
    axs[1].set_xlabel('position in MSA')
    return fig, axs


def __overlapping__(start1, stop1, start2, stop2, extend=0):
    ''' Checks if two segments overlap, or are within a certain distance
        of overlapping, based on "extend". If extend=0, only merges overlaps 
        or directly adjacent domains '''
    c1 = (start1 - start2) >= -extend and (start1 - stop2) <= extend
    c2 = (stop1 - start2) >= -extend and (stop1 - stop2) <= extend
    c3 = (start2 - start1) >= -extend and (start2 - stop1) <= extend
    c4 = (stop2 - start1) >= -extend and (stop2 - stop1) <= extend
    return c1 or c2 or c3 or c4

def __merge_domain_segments__(domains, extend=0):
    ''' Merges domain segments defined as (start,stop) pairs that
        overlapp, or are within <extend> amino acids of overlapping '''
    merged = []
    for i in range(len(domains)):
        start1, stop1 = domains[i]
        merged_last = False; j = 0
        while j < len(merged):
            start2, stop2 = merged[j]
            if __overlapping__(start1, stop1, start2, stop2, extend=extend):
                start3 = min(start1, start2)
                stop3 = max(stop1, stop2)
                merged = merged[:j] + merged[j+1:]
                merged.append((start3, stop3))
                merged_last = True
                break
            else:
                j += 1
        if not merged_last:
            merged.append((start1, stop1))
    if len(merged) < len(domains):
        return __merge_domain_segments__(merged)
    return merged