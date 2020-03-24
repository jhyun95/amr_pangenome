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
import networkx as nx

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
    

def construct_aro_to_drug_network(obo_path):
    '''
    Constructs a DAG from the CARD ARO, such that there exists a path between 
    all AMR genes and their specific impacted drugs. This is acheived by loading 
    all AROs as nodes and adding edges U -> V when:
    1) U "is_a" V (for AMR genes), or reverse V "is_a" U (for drugs)
        - Specific AMR gene inherits resistances of general version of gene
        - Resistance against a drug super class confers resistance to drug members
    2) U has relationship "part_of" or "regulates" V
    3) U has relationship "confers_resistance_to_antibiotic" V
    4) U has relationship "confers_resistance_to_drug_class" V
    5) V has relationship "has_part" U 
        - AMR against a constituent drug contributes to AMR against cotherapy
        
    Parameters
    ----------
    obo_path : str
        Path to CARD ARO network file, usually named aro.obo
    
    Returns
    -------
    G_full : nx.DiGraph
        Networkx DiGraph, such that if an AMR gene contributes to resistance against
        a drug or drug class, there exists a path from the AMR gene ARO to the
        drug/drug class ARO. Nodes are named ARO:#######. Use with G_full.has_path
        to check these relationships, or nx.shortest_path to manually check relationships
    aro_names : dict
        Dictionary mapping ARO:####### to their names
    '''
    
    ''' First identify drug vs gene AROs through connectivity to ARO:1000003 '''
    G_isa = nx.DiGraph(); aro_names = {}
    with open(obo_path, 'r') as f:
        for line in f:
            if line[:8] == 'id: ARO:': # new ARO encountered
                last_aro = line.strip().split()[1]
                G_isa.add_node(last_aro)
            elif line[:5] == 'name:': # name for current ARO encoutnered
                aro_names[last_aro] = line[6:].strip()
            elif line[:5] == 'is_a:': # "is_a" field encountered
                target_aro = line.strip().split()[1]
                G_isa.add_edge(target_aro, last_aro)
            elif line.strip() == '[Typedef]': # end of new ARO terms
                break
    #G_isa.remove_node('ARO:1000001') # root node, links all genes and drugs when present
    drug_aros = nx.descendants(G_isa, 'ARO:1000003') # ARO for “antibiotic molecule”
    drug_aros.add('ARO:1000003')

    ''' Next create the full ontology '''
    G_full = nx.DiGraph()
    valid_relationships = ['part_of', 'regulates', 'confers_resistance_to_antibiotic', 'confers_resistance_to_drug_class']
    with open(obo_path, 'r') as f:
        for line in f:
            if line[:8] == 'id: ARO:': # new ARO encountered
                last_aro = line.strip().split()[1]
                G_full.add_node(last_aro)
            elif line[:5] == 'is_a:': # "is_a" field encountered
                target_aro = line.strip().split()[1]
                if last_aro in drug_aros: # for drugs, build edge drug <- drug superclass
                    G_full.add_edge(target_aro, last_aro)
                else: # for AMR genes, build edge AMR gene -> AMR gene superclass
                    G_full.add_edge(last_aro, target_aro)
            elif line[:13] == 'relationship:': # relationship field encountered
                data = line.split()
                relationship_type = data[1].strip()
                target_aro = data[2]
                if relationship_type in valid_relationships:
                    G_full.add_edge(last_aro, target_aro)
                elif relationship_type == 'has_part':
                    G_full.add_edge(target_aro, last_aro)
            elif line.strip() == '[Typedef]': # end of new ARO terms
                break
    G_full.remove_node('ARO:1000001') # root node, links all genes and drugs when present
    return G_full, aro_names
    
    