#!/usr/bin/env python3

import os
import subprocess
import pandas as pd
from Bio import SeqIO
import re
import tempfile
import sys

sys.path.append('../')
from amr_pangenome import ROOT_DIR  # noqa


class FindJunctions:

    def __init__(self, org, res_dir):

        # get all the required files
        self.__fna_suffix = '_coding_nuc_nr.fna'
        self.__pickle_suffix = '_strain_by_allele.pickle.gz'
        self._org = org
        self.res_dir = res_dir
        self.fa_file = (self.res_dir, self._org)
        self.alleles_file = self._org + self.__pickle_suffix

        # get genes with single alleles, these will be skipped during expensive junction search
        self.single_allele = None
        self.get_single_alleles()

        self._tempdir = tempfile.mkdtemp()
        os.chmod(self._tempdir, 0o755)
        print(self._tempdir)

        self.data_pos = []
        self.junction_row_idx = []
        self.fasta_col_idx = []

    @property
    def fa_file(self):
        return self._fa_file

    @property
    def org(self):
        return self._org

    @property
    def res_dir(self):
        return self.res_dir

    @org.setter
    def org(self, org):
        self._org = org
        self._fa_file = os.path.join(self.res_dir, org + self.__fna_suffix)

    @res_dir.setter
    def res_dir(self, res_dir):
        if not os.path.isdir(res_dir):
            raise NotADirectoryError(f'{res_dir} directory not found. Must pass directory containing '
                                     f'results from pangenome.py.')
        self.res_dir = res_dir
        self._fa_file = os.path.join(res_dir, self._org, self.__fna_suffix)

    @fa_file.setter
    def fa_file(self, directory):
        path, org_name = directory
        fa_path = os.path.join(path, org_name + self.__fna_suffix)
        if not os.path.isfile(fa_path):
            raise FileNotFoundError(f'{fa_path} file not found. Run pangenome.py to generate these files')
        self._fa_file = fa_path

    def get_single_alleles(self):
        df_alleles = pd.read_pickle(os.path.join(self.res_dir, self.alleles_file))
        # drop all genes with only one allele
        allele_freq = {}
        for idx in df_alleles.index:
            gene = re.search(r'_C\d+', idx).group(0).replace('_', '')

            if gene in allele_freq:
                allele_freq[gene] = allele_freq[gene] + 1
            else:
                allele_freq[gene] = 1
        # TODO: double check if this really gets rid of all alleles with one copy
        self.single_allele = [self._org + '_' + i + 'A0' for i in allele_freq if allele_freq[i] == 1]

    def calc_junctions(self, kmer=35, outdir='junctions_out', outname='junctions.csv',
                       outfmt='group'):
        """
        Work horse of the FindJunction class. Its the main method to calculate all the junctions between
        the different alleles from the genes.The calculated junctions are written into the output file.
        Parameters
        ----------
        kmer: int, default 25
            the size of the kmer to use for twopaco.
        outdir: str, default 'junctions_out'
            path to directory where the results will be written
        outname: str, default 'junctions.csv'
            name of the output file
        outfmt: str, {group, gfa2}, default 'group'
            output format for the junctions. only group and gfa2 formats are supported
        """

        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        if not outname.endswith('.csv'):
            outname += '.csv'

        if outfmt not in ['group', 'gfa2']:
            raise ValueError('outfmt must be either \'group\' or \'gfa2\'')

        if outfmt == 'gfa2':
            raise NotImplementedError('This feature is coming to soon. Use \'group\' for outfmt instead')

        # parse the fasta file containing all seqeuences
        parse_fa = SeqIO.parse(self._fa_file, 'fasta')
        rs = next(parse_fa)
        count = 0

        while rs:
            # get the next gene
            gene = re.search(r'_C\d+', rs.id).group(0).replace('_', '')
            if self._org + '_' + gene + 'A0' in self.single_allele:  # skip if gene with one allele
                try:
                    rs = next(parse_fa)
                    continue
                except StopIteration:  # end of file
                    break
            # get all alleles of a gene cluster, then run twopaco and graphdump
            rs = self.group_seq(parse_fa, gene, rs, self._tempdir)
            db_out = self.run_twopaco(self._tempdir, kmer)
            # graph_path = self.run_graphdump(db_out, kmer, outfmt, self._tempdir)
            #
            # # with open(graph_path, 'r') as gfile:
            # #     for lines in gfile.readlines():
            # #         junctions = lines.split(';')

            # TODO: delete all files in the temp folder
            # for testing purposes only
            count += 1
            if count == 1:
                break

    @staticmethod
    def group_seq(fa_generator, gene_name, ref_seq, tmpdir):
        """
        Iterates through the fa_generator object
        created by SeqIO.parse and finds all sequences for the gene cluster i.e.
        those matching the 'gene_name' and writes the sequence and id to temp files."
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

    @staticmethod
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
        two_paco_dir = os.path.join(ROOT_DIR, 'bin/twopaco')
        tp_cmd = [two_paco_dir, '-f', str(kmer), '-o', db_out]
        tp_cmd.extend(fpaths)
        try:
            subprocess.check_output(tp_cmd, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print('Running the TwoPaco command below exited with the following error message:\n')
            print(' '.join(tp_cmd) + '\n')
            print(e.stdout.decode('utf-8'))
            sys.exit(1)
        return db_out

    @staticmethod
    def run_graphdump(db_out, kmer, outfmt, outdir):
        """
        Parse the twopaco output using the compiled graphdump module
        in the bin.
        Parameters
        ----------
        db_out: str
            path to the binary output file from twopaco; generated by run_twopaco
            function
        kmer: int
           kmer size used in the twopaco run that generated the db_out file
        outfmt: str
           output format for the output file; currently only accepts 'group'
        outdir: str
           path to directory where the output file will be written. the output file
           is called 'graphdump.txt'

        Returns
        -------

        """
        gd_dir = os.path.join(ROOT_DIR, 'bin/graphdump')
        gd_cmd = [gd_dir, '-f', outfmt, '-k', str(kmer), db_out]
        graph_path = os.path.join(outdir, 'graphdump.txt')
        with open(graph_path, 'w') as gd_out:
            subprocess.call(gd_cmd, stdout=gd_out, stderr=subprocess.STDOUT)
        return graph_path

    def get_junction_data(self, graphdump_out, fa_list):
        """
        Read the output file of graphdump, and convert it to COO formatted sparse data.
        Parameters
        ----------
        graphdump_out: str, pathlib.Path
             path to the output file of graphdump; generated by run_graphdump with 'group' for
             outfmt
        fa_list: Iterable
            iterable containing fasta file names used in the junctions calculations
        Returns
        -------

        """
        with open(graphdump_out, 'r') as test:
            junction_no = 0
            coo_data = []
            for line in test.readline():
                # each line represents a unique junction
                for junctions in line.split(';'):
                    genome, pos = junctions.split()
                    # this isn't right, this fa_list contains the allele names e.g. C0A21.fna
                    # use allele table to find the fasta files that have that allele and update.
                    fasta = fa_list[genome]
                    fasta = '.'.join(fasta.split('.')[:-1])  # remove the file suffix
                    idx = ('J' + str(junction_no), fasta)
                    coo_data.append([pos, idx])
                junction_no += 1

            return 'asdfasdf'

# main function to run the file
# fj = FindJunctions(org='CC8', res_dir_target='/home/saugat/Documents/CC8_fasta/CDhit_res')
# fj.find_junctions()
# find_junctions(fa_file)
