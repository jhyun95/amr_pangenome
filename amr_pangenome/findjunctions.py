#!/usr/bin/env python3

import os
import subprocess
import pandas as pd
from Bio import SeqIO
import re
import tempfile
import sys
from scipy import sparse
import concurrent.futures
import itertools
import numpy as np

sys.path.append('../')
from amr_pangenome import ROOT_DIR  # noqa


# TODO: write function for removing sentinel and convergent bifurcation junctions
class FindJunctions:

    def __init__(self, org, fna_file, sorted_fna=True):

        # get all the required files
        self._org = org
        self._fna_file = fna_file
        # TODO: if the fasta file is not already sorted, sort now
        if not sorted_fna:
            pass

        # genes with single alleles are skipped during expensive junction search
        self.single_alleles = []
        refseq = SeqIO.parse(self.fna_file, 'fasta')
        allele_names = [rs.id for rs in refseq]
        self.get_single_alleles(allele_names)

    @property
    def fna_file(self):
        return self._fna_file

    @property
    def org(self):
        return self._org

    def get_single_alleles(self, allele_names):
        """
        Updates findjunctions single_allele attribute to add genes with only single allele.
        Parameters
        ----------
        allele_names: iterable
            iterable containing unique allele names
        """
        # drop all genes with only one allele
        allele_freq = {}
        for idx in allele_names:
            gene = re.search(r'_C\d+', idx).group(0).replace('_', '')

            if gene in allele_freq:
                allele_freq[gene] = allele_freq[gene] + 1
            else:
                allele_freq[gene] = 1
        self.single_alleles = [self.org + '_' + i + 'A0' for i in allele_freq if allele_freq[i] == 1]

    def calc_junctions(self, kmer=35, filter_size=34, outname='junctions.csv',
                       outfmt='group', max_processes=1, force=False):
        """
        Workhorse of the FindJunction class. Its the main method to calculate all the junctions between
        the different alleles from the genes.The calculated junctions are written into the output file
        with format unique_junction_name, allele_nucleotide_position.
        Parameters
        ----------
        kmer: int, default 25
            the size of the kmer to use for twopaco. Must be an odd positive integer.
        filter_size: int default 34
            filter size for bloom filter. Larger kmers require more memory. Refer to twopaco
            docs for recom=-cdhnrsumendations based on available memory. WARNING: Raising this value can
            significantly slow down the process by preventing multiprocessing. Read TwoPaco docs
            before messing with it.
        outname: str, path.PATH 'junctions.csv'
            path to the output file
        outfmt: str, {group, gfa2}, default 'group'
            output format for the junctions. only group and gfa2 formats are supported
        max_processes: int, default 1
            maxinum number of processes to use. Note, the number of processes used is limiited
            by the memory usage.
        force: bool, default False
            whether to overwrite the existing output file. If output file already exists and False is passed
            a FileExistsError will be passed.
        """

        if os.path.isfile(outname):
            if force:
                os.remove(outname)
            else:
                raise FileExistsError(f'{outname} exists, pass force=True to overwrite.')

        if not outname.endswith('.csv'):
            outname += '.csv'

        if outfmt == 'gfa2':
            raise NotImplementedError('This feature is coming to soon. Use \'group\' for outfmt instead')

        if outfmt not in ['group', 'gfa2']:
            raise ValueError('outfmt must be either \'group\' or \'gfa2\'')

        if kmer % 2 == 0:
            raise ValueError(f'passed kmer must be and odd number. {kmer} passed instead.')

        # if only one process, write directly to output file
        if max_processes == 1:
            jct_outs = [outname]
        else:  # else have each process write to a temp jct file
            outdir = os.path.dirname(outname)
            jct_outs = [os.path.join(outdir, f'jct{i}.txt') for i in range(max_processes)]
        # use multiprocessing to simulataneously process multiple gene clusters
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
            # map gene cluster with new process
            for jout, gene_cluster in zip(itertools.cycle(jct_outs), self._yield_gene_cluster()):
                gene, tmp_dir = gene_cluster
                f = executor.submit(self._run_single_cluster, tmp_dir, jout, filter_size, gene, kmer, outfmt)
                f.result()

    # TODO: get rid of 'alleles_fna' dir
    def _yield_gene_cluster(self, group=False):
        """
        Private method that generates and yeilds gene clusters from fasta data.
        """
        parse_fa = SeqIO.parse(self.fna_file, 'fasta')
        rs = next(parse_fa)
        count = 0
        while rs:  # iterate through the fasta file
            if count >= 10:
                break
            # get the next gene name, must be formatted as ORG_C000A0 where C is the gene cluster name
            gene = re.search(r'_C\d+', rs.id).group(0).replace('_', '')
            if self.org + '_' + gene + 'A0' in self.single_alleles:
                try:
                    rs = next(parse_fa)
                    continue
                except StopIteration:  # end of file
                    break

            with tempfile.TemporaryDirectory() as tmp_dir:
                fna_temp = os.path.join(tmp_dir, 'alleles_fna')
                os.mkdir(fna_temp)

                # get all alleles of a gene cluster
                rs = self.group_seq(parse_fa, gene, rs, fna_temp, group=group)
                # yield this cluster to a process and move on to the next one
                count += 1
                yield gene, tmp_dir

    def _run_single_cluster(self, tmp_dir, jct_out, filter_size, gene, kmer, outfmt):
        """
           Private method called by 'calc_junctions' to find junctions for a single gene cluster.
           This method finds junctions for all fasta files  in a dir with twopaco and graphjunctions.
           The results are written to the ouput file in jct format. All parameters described here are
           described in the parent 'calc_junctions' function.
        """
        fna_temp = os.path.join(tmp_dir, 'alleles_fna')
        fa_list = os.listdir(fna_temp)
        fpaths = [os.path.join(fna_temp, i) for i in fa_list]
        db_out = self.run_twopaco(fpaths, kmer, filter_size, tmp_dir)
        graph_path = self.run_graphdump(db_out, kmer, outfmt, tmp_dir)

        # gather junctions for the gene junctions and write them to jct formatted file
        junction_list, pos_list = self.get_junction_data(graph_path, fa_list)
        if len(junction_list) != len(pos_list):
            raise AssertionError(f'Number of positions and junctions are not equal for {gene}')
        self.write_jct_file(junction_list, pos_list, jct_out)
        return f'finished {gene}'

    @staticmethod
    def group_seq(fa_generator, gene_name, ref_seq, tmpdir, group=False):
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
        group: bool default False
            Whether to write all the fasta sequences in cluster in a single file

        Returns
        -------
        ref_seq: SeqRecord
            The first SeqRecord object that doesn't contain the 'gene_name'
        """
        while re.search(r'_C\d+', ref_seq.id).group(0).replace('_', '') == gene_name:
            name = ref_seq.id
            faa = ref_seq.seq

            if group:
                fa_loc = os.path.join(tmpdir, gene_name + '.fa')
            else:
                fa_loc = os.path.join(tmpdir, name + '.fa')

            with open(fa_loc, 'a+') as fa:
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
        Read the output file from graphdump, and convert it to jct formatted sparse data.
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

    @staticmethod
    def write_jct_file(junction_list, pos_list, outfile):
        """
        Write the junction names and nucleotide positions into the outfile.
        Format: junctionid, junction_pos.
        """
        with open(outfile, 'a+') as out:
            for jct, pos in zip(junction_list, pos_list):
                out.write(f'{jct},{pos}\n')

    @staticmethod
    def make_junction_strain_df(jct_file, df_alleles, savefile=False,
                                outfile='junction_df.pickle.gz'):
        """
        Creates junction by genome dataframe from jct data and allele by genome data.

        Parameters
        ----------
        jct_file: str, path.PATH
            path to the junction file; junction files have format <jct_number>, <jct_position>
        df_alleles: pandas.DataFrame
            pandas sparse dataframe containing binary alleles by genome data
        savefile: bool, default False
            whether to save jct data by genome dataframe
        outfile: str, path.PATH, default 'junction_df.pickle.gz'
            path to file where the output will be saved if savefile = True.

        Returns
        -------
        jct_df: pd.DataFrame
            sparse dataframe containing junction by genome data. The values are junction positions
            in the gene.
        """

        outdir = os.path.dirname(outfile)
        if not os.path.isdir(outdir):
            raise NotADirectoryError(f'{outdir} does not exist')
        if not outfile.endswith('.gz'):
            outfile += outfile + '.gz'
        junction_list = []
        genomes_list = []
        jpos_list = []

        junction2num = {}
        num2junction = {}
        junction_no = 0
        # all of them have to be a number
        with open(jct_file, 'r') as res_in:
            for line in res_in.readlines():
                name, jpos = line.split(',')
                allele_name = name[:name.rfind('J')]  # clip the junction number
                genomes = df_alleles.loc[allele_name].dropna().index
                genomes_list.extend(genomes)
                jpos_list.extend([int(jpos)] * len(genomes))
                junction_list.extend([name] * len(genomes))

                junction2num.update({name: junction_no})
                num2junction.update({junction_no: name})
                junction_no += 1

        num2genome = dict(zip(range(df_alleles.shape[0]), df_alleles.columns))
        genome2num = dict(zip(df_alleles.columns, range(df_alleles.shape[0])))

        rows = [junction2num[i] for i in junction_list]
        cols = [genome2num[i] for i in genomes_list]

        # convert from COO format to sparse dataframes, and merge duplicate indices
        coo_mat = sparse.coo_matrix((jpos_list, (rows, cols)))
        coo_df = pd.DataFrame.sparse.from_spmatrix(coo_mat).astype(pd.SparseDtype('int64', np.nan))
        coo_df.rename(columns=num2genome, index=num2junction, inplace=True)
        if savefile:
            coo_df.to_pickle(outfile, compression='gzip')
        return coo_df

    def calc_kmer(self, kmers=(31, 33, 35, 37), max_processes=1, sample=1000):
        """
        Sample gene clusters for kmer frequency distribution.

        Parameters
        ----------
        kmers: tuple, iterable default (31, 33, 35, 37)
            kmer sizes to be sampled
        max_processes: int default 1
            number of processes to use
        sample: int default 1000
            number of gene clusters to sample. The first n number of clusters are sampled
        """
        max_freq = []
        sampled = 0
        with tempfile.TemporaryDirectory() as tmp_dir:
            outname = os.path.join(tmp_dir, 'kmer.txt')
            if max_processes == 1:
                kmer_outs = [outname]
            else:  # else have each process write to a temp out file
                kmer_outs = [os.path.join(tmp_dir, f'kmer{i}.txt') for i in range(max_processes)]

            # use multiprocessing to simulataneously process multiple gene clusters
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
                # map gene cluster with new process
                for kout, gene_cluster in zip(itertools.cycle(kmer_outs), self._yield_gene_cluster(group=True)):
                    gene, fa_tmp = gene_cluster
                    flist = os.listdir(fa_tmp + '/alleles_fna/')
                    fa_files = [os.path.join(fa_tmp + '/alleles_fna/', i) for i in flist]
                    f = executor.submit(self._run_ntcard, fa_files, kout, kmers)
                    lfreq = f.result()
                    max_freq.append(lfreq)
                    sampled += 1
                    if sampled >= sample:
                        return max_freq

        return max_freq

    def _run_ntcard(self, fa_file, outfile, kmers=(31, 33, 35, 37)):
        """
        Sample gene clusters for kmer frequency distribution.

        Parameters
        ----------
        fa_file: str, path.PATH or list
            path to fasta file or list of paths to fasta file with the sequnces to be sampled;
            searches for files ending in fna.
        outfile: str, path.PATH
            path to file where the output of ntCard will be written
        kmers: tuple, iterable default (31, 33, 35, 37)
            kmer sizes to be sampled
        Returns
        -------
        lfreq: int
           largest frequency with > 1 kmer
        """

        if type(fa_file) == list:
            fa_file = ' '.join(fa_file)

        # how to sample fasta gene_clusters
        ntcard_dir = os.path.join(ROOT_DIR, 'bin/ntcard')
        kmer_str = ','.join([str(i) for i in kmers])
        nt_cmd = [ntcard_dir, '-t', '1', '-k', kmer_str, '-o', outfile, fa_file]
        try:
            subprocess.call(nt_cmd, stderr=sys.stdout)
        except subprocess.CalledProcessError as e:
            print('Running the ntCard command below exited with the following error message:\n')
            print(' '.join(nt_cmd) + '\n')
            print(e.stdout.decode('utf-8'))
            sys.exit(1)
        return self._calc_max(outfile)

    @staticmethod
    def _calc_max(outfile):
        """
        Reads output of ntCards data and returns the highest kmer frequency with more than one
        appearance.

        Parameters
        ----------
        outfile: str, path.PATH
            path to file containing the output of ntCard

        Returns
        -------
        lfreq: int
            largest frequency with > 1 kmer
        """
        with open(outfile, 'r') as filein:
            lfreq = 1
            lines = filein.readlines()
            lines.reverse()
            # return lines
            for line in lines:
                kmer, freq, n = line.split('\t')
                if int(n) > 1:
                    return int(freq)
            return lfreq


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
