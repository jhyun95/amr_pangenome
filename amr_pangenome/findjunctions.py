#!/usr/bin/env python3

import os
import pandas as pd
from Bio import SeqIO
import re
import tempfile

import sys  # to utilize twopaco and graphdump
sys.path.append('../bin/')

res_dir = '/home/saugat/Documents/CC8_fasta/CDhit_res'
org = 'CC8'
fa_file = os.path.join(res_dir, org + '_coding_nuc_nr.fna')
alleles_file = org + '_strain_by_allele.pickle.gz'
df_alleles = pd.read_pickle(os.path.join(res_dir, alleles_file))

# drop all genes with only one allele
allele_freq = {}
for idx in df_alleles.index:
    gene = re.search(r'_C\d+', idx).group(0).replace('_', '')

    if gene in allele_freq:
        allele_freq[gene] = allele_freq[gene] + 1
    else:
        allele_freq[gene] = 1

# TODO: double check if this really gets rid of all allles with one copy
single_allele = [org + '_' + i + 'A0' for i in allele_freq if allele_freq[i] == 1]


# gotta fix it so that its assigning the unique allele name to fasta
def group_seq(fa_generator, gene_name, ref_seq, tmpdir):
    """
    Iterates through the fa_generator object created by SeqIO.parse and finds
    all sequences for the gene cluster i.e. those matching the 'gene_name' and
    writes the sequence and id to temp files."
    Parameters
    ----------
    fa_generator: Generator
        Generator object created by passing fasta file to SeqIO.parse from Bio
        package.
    gene_name: str
        Name of the gene in the the gene cluster. Has the format r'C\d+'
    ref_seq: SeqRecord
        The current SeqRecord object yielded by fa_generator
    tmpdir: str
        Path to the temporary directory where the fasta file will be stored

    Returns
    -------
    ref_seq: SeqRecord
        The first SeqRecord object that doesn't contain the 'gene_name'
    """
    while re.search(r'_C\d+', ref_seq.id).group(0).replace('_', '') == gene_name:
        name = ref_seq.id
        faa = ref_seq.seq
        fa_loc = os.path.join(tmpdir, name + '.fa')
        with open(fa_loc, 'w') as fa:
            fa.write('>{}\n{}\n'.format(name, faa))

        try:
            ref_seq = next(fa_generator)
        except StopIteration:
            return
    return ref_seq


parse_fa = SeqIO.parse(fa_file, 'fasta')
rs = next(parse_fa)
count = 0
# tmpdir to save the interim fasta files
tempdir = tempfile.mkdtemp()
print(tempdir)
while rs:
    # get the next
    gene = re.search(r'_C\d+', rs.id).group(0).replace('_', '')
    print(gene)
    if org + '_' + gene + 'A0' in single_allele:  # skip if gene with one allele
        try:
            rs = next(parse_fa)
            continue
        except StopIteration:  # end of file
            break

    rs = group_seq(parse_fa, gene, rs, tempdir)

    # TODO: run twopaco and graphdump here
    # TODO: delete all files in the temp folder
    # for testing purposes only
    count += 1
    if count == 2:
        break
