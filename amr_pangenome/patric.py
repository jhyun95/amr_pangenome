#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 18:55:10 2020

@author: jhyun95

Tools for downloading and validating data through PATRIC's FTP server.
"""

import os, urllib
import pandas as pd

VALID_PATRIC_FILES = ['faa','features.tab','ffn','frn','gff','pathway.tab',
                      'spgene.tab','subsystem.tab','fna']

def download_patric_genomes(genomes, output_dir, filetypes=['fna','faa','gff','spgene.tab'], redownload=False):
    '''
    Download data associated with a list of PATRIC genomes.
    
    Parameters
    ----------
    genomes : list
        List of strings containing PATRIC genome IDs to download
    output_dir : str
        Path to directory to save genomes. Will create a subfolder 
        for each genome in this directory.
    filetypes : str
        List of PATRIC genome-specific files to download per genome. 
        Valid options include 'faa', 'features.tab', 'ffn', 'frn',
        'gff', 'pathway.tab', 'spgene.tab', 'subsystem.tab', and
        'fna'. 'PATRIC' in filename is dropped automatically.
        See ftp://ftp.patricbrc.org/genomes/<genome id>/ for 
        examples (default ['fna','faa','gff','spgene.tab'])
    redownload : bool
        If True, re-downloads files that exist locally (default False)
    '''
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    ''' Process filetypes '''
    source_target_filetypes = [] 
    for ftype in filetypes:
        if ftype in VALID_PATRIC_FILES: # valid file type
            ftype_source = 'PATRIC.' + ftype if ftype != 'fna' else ftype # all files except FNA preceded by 'PATRIC'
            ftype_target = ftype # drop 'PATRIC' in output files
            source_target_filetypes.append( (ftype_source, ftype_target) )
        elif ftype.replace('PATRIC.','') in VALID_PATRIC_FILES: # valid file type without PATRIC label
            ftype_source = ftype # keep 'PATRIC' for downloading files
            ftype_target = ftype.replace('PATRIC.','') # drop 'PATRIC' in output files
            source_target_filetypes.append( (ftype_source, ftype_target) )
        else: # invalid filetype
            print 'Invalid filetype:', ftype
    
    ''' Download relevant files '''
    for i, genome in enumerate(genomes):
        ''' Set up source and target locations '''
        genome_source = 'ftp://ftp.patricbrc.org/genomes/' + genome + '/' + genome # base link to genome files
        genome_dir = output_dir + '/' + genome + '/' # genome-specific output directory
        genome_dir = genome_dir.replace('//','/')
        genome_target = genome_dir + genome # genome-specific output base filename
        if not os.path.exists(genome_dir):
            os.mkdir(genome_dir)

        ''' Process individual files '''
        bad_genomes = []
        try:
            for source_filetype, target_filetype in source_target_filetypes:
                source = genome_source + '.' + source_filetype
                target = genome_target + '.' + target_filetype
                if os.path.exists(target) and not redownload:
                    print i+1, 'Already exists:', target
                else:
                    print i+1, source, '->', target
                    urllib.urlretrieve(source, target)
                    urllib.urlcleanup()
        except IOError: # genome ID not found
            print 'Bad genome ID:', genome
            os.rmdir(genome_dir)
            bad_genomes.append(genome)
    return bad_genomes
   

def validate_patric_genomes(genomes_dir, summary_file='data/PATRIC/PATRIC_genome_summary.tsv'):
    '''
    Some validation tests to check whether PATRIC genomes downloaded correctly.
    Applies specifically to FNA, FAA, GFF, and SPGENE.TAB files.
    1) Check GFF CDS count == PATRIC summary CDS count. This will not be exact due 
        to more specific features such as transcripts, but should be at least 98% same.
    2) Check GFF CDS count == FAA CDS count. These should be equal.
    3) Check FNA contig count == PATRIC summary contig count. These should be equal.
    4) Check that SPGENE.TAB is not empty
    
    Parameters
    ----------
    genomes_dir : str
        Directory with genome subfolders as used with download_patric_genomes().
        Will check that GFF, FNA, FAA, and SPGENE.TAB exists before testing.
    summary_file : str
        Path to genome summary file. Refers to:
        ftp://ftp.patricbrc.org/RELEASE_NOTES/archive/Dec2019/genome_summary
    '''
    
    ''' Load summary file '''
    df_summary_full = pd.read_csv(summary_file, sep='\t', dtype={'genome_id':str})
    df_summary_full = df_summary_full.set_index('genome_id')
    
    ''' Evaluate consistency between summary and downloaded files '''
    for genome in os.listdir(genomes_dir):
        genome_specific_dir = genomes_dir + '/' + genome + '/'
        genome_specific_dir = genome_specific_dir.replace('//','/')
        gff_file = genome_specific_dir + genome + '.gff'
        fna_file = genome_specific_dir + genome + '.fna'
        faa_file = genome_specific_dir + genome + '.faa'
        spgene_file = genome_specific_dir + genome + '.spgene.tab'
        files_present = map(lambda x: os.path.exists(x), [gff_file, fna_file, faa_file, spgene_file])
        files_present = reduce(lambda x,y: x and y, files_present)
        
        if files_present:
            print 'Testing', genome, genome_specific_dir
            patric_contigs = df_summary_full.loc[genome,'contigs']
            patric_cds = df_summary_full.loc[genome,'patric_cds']
        
            ''' Checking GFF against contigs and CDS '''
            with open(gff_file, 'r') as f_gff:
                gff_contigs = set(); gff_cds = set()
                for line in f_gff:
                    if line[0] != '#':
                        data = line.split('\t')
                        if len(data) > 3:
                            contig, src, ftype =  data[:3]
                            gff_contigs.add(contig)
                            is_cds = (ftype == 'CDS') and ('fig|' + genome + '.peg' in line)
                            if is_cds:
                                fname = data[-1].split(';')[0].split('|')[-1]
                                gff_cds.add(fname)

            ''' Checking FNA against contigs '''
            with open(fna_file, 'r') as f_fna:
                fna_contigs = 0
                for line in f_fna:
                    if line[0] == '>':
                        fna_contigs += 1

            ''' Checking FAA against GFF CDS '''
            with open(faa_file, 'r') as f_faa:
                faa_cds = set()
                for line in f_faa:
                    if line[0] == '>':
                        fname = line.split()[0].split('|')[1]
                        faa_cds.add(fname)

            ''' Checking spgene counts '''
            df_spgene = pd.read_csv(spgene_file, sep='\t')

            cds_accuracy = round(100.0*len(gff_cds) / patric_cds , 2) 
            contig_check2 = fna_contigs == patric_contigs
            cds_check1 = len(faa_cds) == len(gff_cds)
            cds_check2 = cds_accuracy > 98.0
            spgene_check = df_spgene.shape[0] > 0
            print '\tSpecial genes:', df_spgene.shape[0]

            if not(contig_check2 and cds_check1 and cds_check2):
                print '\tFNA contig count match:', contig_check2, fna_contigs, patric_contigs
                print '\tFAA/GFF CDS count match:', cds_check1, len(faa_cds), len(gff_cds)
                #print '\t', gff_cds.difference(faa_cds)
                print '\tGFF CDS accuracy:', cds_check2, cds_accuracy
                
                
                