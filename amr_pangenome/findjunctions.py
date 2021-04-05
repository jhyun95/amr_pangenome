#!/usr/bin/env python3

import os
import subprocess
import pandas as pd
from Bio import SeqIO
import re
import tempfile
import sys
from scipy import sparse

sys.path.append('../')
from amr_pangenome import ROOT_DIR  # noqa


class FindJunctions:

    def __init__(self, org, res_dir):
        self.__fna_suffix = '_coding_nuc_nr.fna'
        self.__pickle_suffix = '_strain_by_allele.pickle.gz'

        # get all the required files
        self.org = org
        self.res_dir = res_dir
        self.fa_file = os.path.join(self.res_dir, self._org + self.__fna_suffix)
        self.alleles_file = self._org + self.__pickle_suffix

        # genes with single alleles are skipped during expensive junction search
        self.single_alleles = []
        df_alleles = pd.read_pickle(os.path.join(self.res_dir, self.alleles_file))
        self.get_single_alleles(df_alleles)

        self.pos_data = []  # nucleotide position data for  junctions
        self.junction_row_idx = []  # junction names

    @property
    def org(self):
        return self._org

    @org.setter
    def org(self, org):
        self._org = org

    @property
    def res_dir(self):
        return self._res_dir

    @res_dir.setter
    def res_dir(self, res_dir):
        if not os.path.isdir(res_dir):
            raise NotADirectoryError(f'{res_dir} directory not found. Must pass directory containing '
                                     f'results from pangenome.py.')
        self._res_dir = res_dir
        self._fa_file = os.path.join(res_dir, self._org, self.__fna_suffix)

    @property
    def fa_file(self):
        return self._fa_file

    @fa_file.setter
    def fa_file(self, fa_path):
        if not os.path.isfile(fa_path):
            raise FileNotFoundError(f'{fa_path} file not found. Run pangenome.py to generate these files')
        self._fa_file = fa_path

    def get_single_alleles(self, df_alleles):
        # drop all genes with only one allele
        allele_freq = {}
        for idx in df_alleles.index:
            gene = re.search(r'_C\d+', idx).group(0).replace('_', '')

            if gene in allele_freq:
                allele_freq[gene] = allele_freq[gene] + 1
            else:
                allele_freq[gene] = 1
        self.single_alleles = [self.org + '_' + i + 'A0' for i in allele_freq if allele_freq[i] == 1]

    def calc_junctions(self, kmer=25, filter_size=36,  outdir='junctions_out', outname='junctions.csv',
                       outfmt='group', force=False):
        """
        Workhorse of the FindJunction class. Its the main method to calculate all the junctions between
        the different alleles from the genes.The calculated junctions are written into the output file
        with format unique_junction_name, allele_nucleotide_position.
        Parameters
        ----------
        kmer: int, default 25
            the size of the kmer to use for twopaco. Must be an odd positive integer.
        filter_size: int default 36
            filter size for bloom filter. Larger kmers require more memory. Refer to twopaco
            docs for recommendations based on available memory. Default 36 is recommended for
            16GB of memory.
        outdir: str, default 'junctions_out'
            path to directory where the results will be written
        outname: str, default 'junctions.csv'
            name of the output file
        outfmt: str, {group, gfa2}, default 'group'
            output format for the junctions. only group and gfa2 formats are supported
        force: bool, default False
            whether to overwrite the existing output file. If output file already exists and False is passed
            a FileExistsError will be passed.
        """

        coo_out = os.path.join(outdir, 'coo.txt')
        if os.path.isfile(coo_out):
            if force:
                os.remove(coo_out)
            else:
                raise FileExistsError(f'{coo_out} exists, pass force=True to overwrite.')

        if outfmt == 'gfa2':
            raise NotImplementedError('This feature is coming to soon. Use \'group\' for outfmt instead')

        if outfmt not in ['group', 'gfa2']:
            raise ValueError('outfmt must be either \'group\' or \'gfa2\'')

        if kmer % 2 == 0:
            raise ValueError(f'passed kmer must be and odd number. {kmer} passed instead.')

        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        if not outname.endswith('.csv'):
            outname += '.csv'

        # parse the fasta file containing all seqeuences
        parse_fa = SeqIO.parse(self.fa_file, 'fasta')
        rs = next(parse_fa)
        count = 0

        while rs:
            # get the next gene
            gene = re.search(r'_C\d+', rs.id).group(0).replace('_', '')
            if self.org + '_' + gene + 'A0' in self.single_alleles:  # skip if gene with one allele
                try:
                    rs = next(parse_fa)
                    continue
                except StopIteration:  # end of file
                    break
            # take this val and split it into multiprocess
            with tempfile.TemporaryDirectory() as tmp_dir:
                fna_temp = os.path.join(tmp_dir, 'alleles_fna')
                os.mkdir(fna_temp)

                # get all alleles of a gene cluster, then run twopaco and graphdump
                rs = self.group_seq(parse_fa, gene, rs, fna_temp)
                fa_list = os.listdir(fna_temp)
                fpaths = [os.path.join(fna_temp, i) for i in fa_list]
                db_out = self.run_twopaco(fpaths, kmer, filter_size, tmp_dir)
                graph_path = self.run_graphdump(db_out, kmer, outfmt, tmp_dir)

                # gather junctions for the gene junctions and write them to coo formatted file
                junction_list, pos_list = self.get_junction_data(graph_path, fa_list)
                if len(junction_list) != len(pos_list):
                    raise AssertionError(f'Number of positions and junctions are not equal for {gene}')
                self.write_coo_file(junction_list, pos_list, coo_out)

            # for testing purposes only
            count += 1
            if count >= 1:
                break

    # TODO: gene by allele fasta file
    # TODO: Make sure to include single allele genes
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
            Name of the gene in the the gene cluster. Has the format 'C\\d+'
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

    # TODO: change outdir to outpath
    @staticmethod
    def run_twopaco(fpaths, kmer, filtersize, outdir):
        """

        Parameters
        ----------
        fpaths: list
            list of paths to fasta files
        kmer: int
            size of the kmers to be used for twopaco junction finder
        filtersize: int default 36
            filter size for bloom filter. Larger kmers require more memory. Refer to twopaco
            docs for recommendations based on available memory. Default 36 is recommended for
        outdir: str or path.PATH
            path to the output directory where the output will be stored
        Returns
        -------
        db_out: str
            path to the file of containing the output of twopaco
        """

        db_out = os.path.join(outdir, 'debrujin.bin')
        two_paco_dir = os.path.join(ROOT_DIR, 'bin/twopaco')
        tp_cmd = [two_paco_dir, '-k', str(kmer), '-f', str(filtersize), '-o', db_out, '-t', '8']
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

    @staticmethod
    def get_junction_data(graphdump_out, fa_list):
        """
        Read the output file from graphdump, and convert it to COO formatted sparse data.
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

        fa_list = [os.path.splitext(i)[0] for i in fa_list]
        junction_no = 0
        pos_data = []
        junction_row_idx = []
        with open(graphdump_out, 'r') as graph_in:
            for line in graph_in.readlines():
                # each line represents a unique junction and the allele
                for junction in line.split(';')[:-1]:  # last entry is \n
                    allele, pos = junction.split()
                    junction_id = fa_list[int(allele.strip())] + 'J' + str(junction_no)
                    pos_data.append(pos)
                    junction_row_idx.append(junction_id)
                junction_no += 1

        return junction_row_idx, pos_data

    def make_junction_strain_df(self, df_alleles):
        df_junction_strain = pd.DataFrame(index=self.junction_row_idx, columns=df_alleles.columns,
                                          dtype=int)

        return df_junction_strain

    @staticmethod
    def write_coo_file(junction_list, pos_list, outfile):
        """
        Write the junction names and nucleotide positions into the outfile
        """
        with open(outfile, 'a+') as out:
            for jct, pos in zip(junction_list, pos_list):
                out.write(f'{jct},{pos}\n')
