#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 18:43:05 2020

@author: jhyun95

Scripts for managing AMR annotation (not analysis), primarily 
a wrapper for CARD's RGI tool.
"""

import subprocess as sp
import os

import pandas as pd
import numpy as np

def run_rgi(fasta_in, rgi_out, rgi_args={'-a':'DIAMOND', '-n':1}, clean_headers=True):
    ''' 
    RGI wrapper for FNA (contig) or FAA (protein) files. Optionally cleans
    files up (needed for some PATRIC FAA files where headers contain
    characters incompatible with json importing).
    
    Parameters
    ----------
    fasta_in : str
        Path to FNA or FAA file to be annotated.
    rgi_out : str
        Path to output RGI files. Recommended that this is in its own folder
    rgi_args : dict
        Additional args for rgi main, other than -i, -o and -t.
        (default {'-a':'DIAMOND', '-n':1})
    clean_headers : bool
        If true, creates a copy of the import fasta with shortened headers
        (i.e. header.split()[0] that may resolve some RGI import issues (default True).
    '''
    if clean_headers: # shorten headers in fasta, create *.tmp copy of fasta  
        fasta_tmp = fasta_in + '.tmp'
        with open(fasta_in, 'r') as f_in:
            with open(fasta_tmp, 'w+') as f_out:
                for line in f_in:
                    output = line.split()[0] if line[0] == '>' else line
                    f_out.write(output.strip() + '\n')
        fasta = fasta_tmp
    else: # use original file
        fasta = fasta_in
    
    mode = 'contig' if fasta_in[-4:].upper() == '.FNA' else 'protein'
    args = ['rgi', 'main', '-i', fasta, '-o', rgi_out, '-t', mode]
    for key, value in rgi_args.items():
        args += [key, str(value)]
    print ' '.join(args)
    print sp.check_output(args)
    
    if clean_headers: # delete temporary fasta
        os.remove(fasta_tmp)
        
def build_resistome(rgi_txt, aro_out, df_alleles, map_aros=True, skip_loose=True):
    '''
    Processes the generated txt file after running RGI on a non-redundant
    CDS pan-genome, as described by df_alleles. Yields a binary
    ARO x genome table, with AROs correspondings to alleles detected by RGI.
    
    Parameters
    ----------
    rgi_txt : str
        Path to RGI outputted text file
    aro_out : str
        Path to output ARO x genome table, as CSV or CSV.GZ
    df_alleles : str or pd.DataFrame
        Dataframe of allele x genome table, or path to the table
    map_aros : bool
        If True, renames alleles to AROs and combines same-ARO alleles (default True) 
    skip_loose : bool
        If True, skips Loose hits from RGI (default True)
        
    Returns
    df_rgi : pd.DataFrame
        Dataframe containing raw RGI txt output, optionally with Loose hits removed
    df_aro : pd.DataFrame
        Dataframe with binary ARO x genome table
    '''
    
    ''' Load RGI hits '''
    df_rgi = pd.read_csv(rgi_txt, sep='\t')
    if skip_loose:
        df_rgi = df_rgi[df_rgi.Cut_Off != 'Loose']
    
    ''' Map RGI alleles to AROs '''
    df = df_rgi.loc[:,['ORF_ID', 'ARO']]
    allele_to_aro = {}
    for row in df.itertuples(name=None):
        i, allele, aro = row
        if allele in allele_to_aro: # allele mapped to multiple AROs
            print 'Duplicate hit:', allele
        allele_to_aro[allele] = aro
        
    ''' Slice df_alleles to just RGI hits '''
    aro_alleles = sorted(allele_to_aro.keys())
    if type(df_alleles) == str:
        df_aro = pd.read_csv(df_alleles, index_col=0).loc[aro_alleles,:]
    else: # allele x genome table provided directly
        df_aro = df_alleles.loc[aro_alleles,:]
    
    ''' Optionally convert alleles to AROs '''
    if map_aros:
        df_aro.index = df_aro.index.map(lambda x: allele_to_aro[x]) # map alleles to AROs
        df_aro = df_aro.fillna(0).groupby([df_aro.index]).sum() > 0 # boolean table
        df_aro = df_aro.replace({True:1, False:np.nan})
    df_aro.to_csv(aro_out)
    
    return df_rgi, df_aro
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
