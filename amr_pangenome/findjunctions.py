#!/usr/bin/env python3

import os
import subprocess
import pandas as pd
from Bio import SeqIO
import re
import tempfile
import sys

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


def find_junctions(fasta, kmer=35, outdir='junctions_out', outname='junctions.csv',
                   outfmt='group'):

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    if not outname.endswith('.csv'):
        outname += '.csv'

    if outfmt not in ['group', 'gfa2']:
        raise ValueError('outfmt must be either \'group\' or \'gfa2\'')

    if outfmt == 'gfa2':
        raise NotImplementedError('This feature is coming to soon. Use \'group\' for outfmt instead')

    tempdir = tempfile.mkdtemp()
    os.chmod(tempdir, 0o755)
    print(tempdir)

    # parse the fasta file containing all seqeuences
    parse_fa = SeqIO.parse(fasta, 'fasta')
    rs = next(parse_fa)
    count = 0

    while rs:
        # get the next gene
        gene = re.search(r'_C\d+', rs.id).group(0).replace('_', '')
        if org + '_' + gene + 'A0' in single_allele:  # skip if gene with one allele
            try:
                rs = next(parse_fa)
                continue
            except StopIteration:  # end of file
                break
        # get all alleles of a gene cluster, then run twopaco and graphdump
        rs = group_seq(parse_fa, gene, rs, tempdir)
        db_out = run_twopaco(tempdir, kmer)
        graph_path = run_graphdump(db_out, kmer, outfmt, tempdir)

        with open(graph_path, 'r') as gfile:
            for lines in gfile.readlines():
                junctions = lines.split(';')

        # TODO: Reformat the graphdump output into junction by genome, should be new function
        # TODO: delete all files in the temp folder

        # for testing purposes only
        count += 1
        if count == 1:
            break


def run_graphdump(db_out, kmer, outfmt, tempdir):
    gd_cmd = ['../bin/graphdump', '-f', outfmt, '-k', str(kmer), db_out]
    graph_path = os.path.join(tempdir, 'graphdump.txt')
    with open(graph_path, 'w') as gd_out:
        subprocess.call(gd_cmd, stdout=gd_out, stderr=subprocess.STDOUT)
    return graph_path


def run_twopaco(tempdir, kmer):
    """

    Parameters
    ----------
    tempdir: str
        temporary directory where all the fasta files are stored
    kmer: int
        size of the kmers to be used for twopaco junction finder

    Returns
    -------
    db_out: str
        path to the file of containing the output of twopaco
    """
    fa_list = os.listdir(tempdir)
    fpaths = [os.path.join(tempdir, i) for i in fa_list]
    db_out = os.path.join(tempdir, 'debrujin.bin')
    tp_cmd = ['../bin/twopaco', '-f', str(kmer), '-o', db_out]
    tp_cmd.extend(fpaths)
    try:
        subprocess.check_output(tp_cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print('Running the TwoPaco command below exited with the following error message:\n')
        print(' '.join(tp_cmd) + '\n')
        print(e.stdout.decode('utf-8'))
        sys.exit(1)
    return db_out


# main function to run the file
find_junctions(fa_file)
