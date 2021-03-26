#!/usr/bin/env python3

import sys
sys.path.append('../amr_pangenome/')
sys.path.append('..')

import amr_pangenome
import os

org_dir = '/home/saugat/Documents/CC8_fasta/'
genome_faas = []
for fname in os.listdir(org_dir):
    if 'CC8' in fname:
#         print(fname)
        continue
    if os.path.isdir(os.path.join(org_dir,  fname)):
        genome_faas.append(org_dir + fname + '/' + fname + '.faa')

cdhit_args = {'-n':5, '-c':0.8, '-aL':0.8, '-T':10}

genomes_gff_fna = []
for genome in sorted(os.listdir(org_dir)):
    if os.path.isdir(org_dir + genome):
        gff = org_dir + genome + '/' + genome + '.gff'
        fna = org_dir + genome + '/' + genome + '.fna'
        if os.path.exists(gff) and os.path.exists(fna):
            genomes_gff_fna.append((gff,fna))
            


test_gff_fna = genomes_gff_fna[0:20]

output_dir = '/home/saugat/Documents/CC8_fasta/CDhit_res'
df_alleles, df_genes = amr_pangenome.pangenome.build_cds_nucl_pangenome(
    genome_data=genomes_gff_fna, 
    output_dir=output_dir, name='CC8', 
    cdhit_args=cdhit_args, save_csv=False,
    fastasort_path='/home/saugat/bin/fastasort')


