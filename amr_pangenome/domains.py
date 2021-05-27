#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Thu Wed 26 23:34:17 2021

@author: jhyun95

"""

import numpy as np
import pandas as pd

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