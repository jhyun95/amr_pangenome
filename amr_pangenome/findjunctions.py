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
import fileinput
from amr_pangenome import ROOT_DIR  # noqa
from amr_pangenome.mlgwas import sparse_arrays_to_sparse_matrix


# TODO: write function for removing sentinel and convergent bifurcation junctions
class FindJunctions:

    def __init__(self, org, fna_file, allele_file, sorted_fna=True):
        """
        Object to keep track of all the data files needed to calculate junctions from allele by genome data.

        Parameters
        ----------
        org: str
            name of the organism to be used in allele and junction name
        fna_file: str or path.PATH
            path to fasta file containing the nuc sequence of the gene clusters
        allele_file: str or path.PATH
            path to the file containing the allele by genome dataframe; must be pickled data
        sorted_fna: bool
           whether fasta file is already sorted. If false, the fasta file will be sorted by sequence id. Default True
        """

        # get all the required files
        self._org = org
        self._fna_file = fna_file
        self._allele_file = allele_file

        # initialize other attributes that will be filled in later
        # TODO: make getters and setters for these attributes so they can't be messed with.
        self.df_alleles = None
        self.uniq_genomes = None
        self.alleles = None
        self.genomes = None
        self.df_jct = None
        self.jct_path = None
        self.junctions2num = None

        # TODO: if the fasta file is not already sorted, sort now
        if not sorted_fna:
            raise NotImplementedError('Coming soon')

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

    @property
    def allele_file(self):
        return self._allele_file

    def get_single_alleles(self, allele_names):
        """
        Updates findjunctions single_allele attribute to add genes with only single allele.
        Parameters
        ----------
        allele_names: iterable
            iterable containing unique allele names
        """
        # drop all genes with only one allele
        genes = [re.search(r'_C\d+', i).group(0).replace('_', '') for i in allele_names]
        duplicate = {}
        for idx in genes:
            duplicate[idx] = idx in duplicate
        self.single_alleles = [f'{self.org}_{i}A0' for i in duplicate if not duplicate[i]]

    # TODO the temporary jct files should be written in the tempdirectory, need to figure out how to implement
    # that test before we can add that here.
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

        processes = []
        # use multiprocessing to simulataneously process multiple gene clusters
        with tempfile.TemporaryDirectory() as tmp_dir:
            tp_outs = [os.path.join(tmp_dir, f'tpout{i}/') for i in range(max_processes)]
            [os.makedirs(i) for i in tp_outs]
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
                # map gene cluster with new process
                for pn, gene_cluster in zip(itertools.cycle(np.arange(max_processes)),
                                            self._yield_gene_cluster(tmp_dir)):
                    jout, tp_out = jct_outs[pn], tp_outs[pn]
                    gene, fpaths = gene_cluster
                    f = executor.submit(self._run_single_cluster, fpaths, jout,
                                        filter_size, gene, kmer, outfmt, tp_out)
                    processes.append(f)
                for futures in concurrent.futures.as_completed(processes):
                    futures.result()

        if max_processes == 1:
            return
        # Cat all the files from mp together into the final output
        jct_exists = [i for i in jct_outs if os.path.isfile(i)]
        with open(outname, 'w') as fout, fileinput.input(jct_exists) as fin:
            for line in fin:
                fout.write(line)
        [os.remove(i) for i in jct_exists]

    def _yield_gene_cluster(self, direct, group=False):
        """
        Private method that generates and yeilds gene clusters from fasta data.

        Parameters
        ----------
        direct: str, path.PATH
              path to the directory where the fasta files will be saved
        group: bool default False
              whether to group the sequnces into a single fasta file
        Yields
        ------
        rs: Bio.SeqRecord
            the SeqRecord of the fasta file with pointer at the end of the gene cluster
        fa_locs: list
            list of fasta file(s) where the sequences of the genes in the gene cluster were written
        """
        parse_fa = SeqIO.parse(self.fna_file, 'fasta')
        rs = next(parse_fa)
        while rs:  # iterate through the fasta file
            # get the next gene name, must be formatted as ORG_C000A0 where C is the gene cluster name
            gene = re.search(r'_C\d+', rs.id).group(0).replace('_', '')
            if self.org + '_' + gene + 'A0' in self.single_alleles:
                try:
                    rs = next(parse_fa)
                    continue
                except StopIteration:  # end of file
                    break
            # get all alleles of a gene cluster
            cluster_res = self.group_seq(parse_fa, gene, rs, direct, group=group)
            rs, fa_locs = cluster_res
            # yield this cluster to a process and move on to the next one
            yield gene, fa_locs

            if rs is None:  # end of file
                return 0

    def _run_single_cluster(self, fpaths, jct_out, filter_size, gene, kmer, outfmt, outdir):
        """
           Private method called by 'calc_junctions' to find junctions for a single gene cluster.
           This method finds junctions for all fasta files  in a dir with twopaco and graphjunctions.
           The results are written to the ouput file in jct format. All parameters described here are
           described in the parent 'calc_junctions' function.
        """
        db_out = self.run_twopaco(fpaths, kmer, filter_size, outdir)
        gd_out = self.run_graphdump(db_out, kmer, outfmt)
        # gather junctions for the gene junctions and write them to jct formatted file
        junction_list, pos_list = self.get_junction_data(gd_out, fpaths)
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
        fa_locs = []
        write_mode = 'w'
        if group:
            fa_locs.append(os.path.join(tmpdir, gene_name + '.fa'))
            write_mode = 'a+'
        while re.search(r'_C\d+', ref_seq.id).group(0).replace('_', '') == gene_name:
            name = ref_seq.id
            faa = ref_seq.seq

            if group:
                fa_loc = fa_locs[0]
            else:
                fa_loc = os.path.join(tmpdir, name + '.fa')
                fa_locs.append(fa_loc)

            with open(fa_loc, write_mode) as fa:
                fa.write('>{}\n{}\n'.format(name, faa))

            try:
                ref_seq = next(fa_generator)
            except StopIteration:  # this will probably break the code
                ref_seq = None
                break
        return ref_seq, fa_locs

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
        with tempfile.TemporaryDirectory() as tp_temp:
            tp_cmd = [two_paco_dir, '-k', str(kmer), '-f', str(filtersize),
                      '-o', db_out, '-t', '1', '--tmpdir', str(tp_temp)]
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
    def run_graphdump(db_out, kmer, outfmt):
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
        Returns
        -------
        gd_out: str
            captured stdout of the the graphdump call
        """

        gd_dir = os.path.join(ROOT_DIR, 'bin/graphdump')
        gd_cmd = [gd_dir, '-f', outfmt, '-k', str(kmer), db_out]
        gd_out = subprocess.check_output(gd_cmd, stderr=subprocess.STDOUT)
        return gd_out.decode('utf-8')  # convert to string

    @staticmethod
    def get_junction_data(graphdump_out, fa_list):
        """
        Read the output file from graphdump, and convert it to jct formatted sparse data.
        Parameters
        ----------
        graphdump_out: str
             captured stdout of the the graphdump call; generated by run_graphdump with 'group' for
             outfmt
        fa_list: Iterable
            iterable containing fasta file names used in the junctions calculations
        Returns
        -------

        """
        fa_list = [os.path.splitext(os.path.basename(i))[0] for i in fa_list]
        junction_no = 0
        pos_data = []
        junction_row_idx = []
        for line in [ln for ln in graphdump_out.splitlines() if ln]:  # remove blank lines
            # each line represents a unique junction and the allele
            for junction in line.split(';')[:-1]:  # last entry is \n
                allele, pos = junction.split()
                try:
                    junction_id = fa_list[int(allele.strip())] + 'J' + str(junction_no)
                except IndexError:
                    print("RAN INTO ERROR:")
                    print(fa_list)
                    return ['jct_err'], [-1]

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

    def calc_kmer(self, kmers=(31, 33, 35, 37), max_processes=1, sample=10e6):
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
        Return
        ------
        max_freq: list
           list of maximum number of times a kmer is repeated in a gene cluster; length of
           the list is equal to the number of gene clusters or sample parameter, whichever
           is lower.
        """
        max_freq = []
        sampled = 0
        with tempfile.TemporaryDirectory() as tmp_dir:
            outname = os.path.join(tmp_dir, 'kmer.txt')
            if max_processes == 1:
                kmer_outs = [outname]
            else:  # else have each process write to a temp out file
                kmer_outs = [os.path.join(tmp_dir, f'kmer{i}.txt') for i in range(max_processes)]

            processes = []
            # use multiprocessing to simulataneously process multiple gene clusters
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
                # map gene cluster with new process
                for kout, gene_cluster in zip(itertools.cycle(kmer_outs),
                                              self._yield_gene_cluster(tmp_dir, group=True)):
                    gene, fpaths = gene_cluster  # returns gene name and path to fa files
                    f = executor.submit(self._run_ntcard, fpaths, kout, kmers)
                    processes.append(f)
                    sampled += 1
                    if sampled >= sample:
                        break
                for futures in concurrent.futures.as_completed(processes):
                    lfreq = futures.result()
                    if lfreq == -1:  # skip results from clusters that create rare ntcard bugs.
                        continue
                    max_freq.append(lfreq)

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
        # TODO: figure out how to prevent ntcard from writing to stdout
        ntcard_dir = os.path.join(ROOT_DIR, 'bin/ntcard')
        kmer_str = ','.join([str(i) for i in kmers])
        nt_cmd = [ntcard_dir, '-t', '1', '-k', kmer_str, '-o', outfile, fa_file]
        try:
            subprocess.check_call(nt_cmd, shell=False, stdout=subprocess.DEVNULL)
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
            lfreq = 0
            lines = filein.readlines()
            lines.reverse()
            # return lines
            for line in lines:
                try:
                    kmer, freq, n = line.split('\t')
                except ValueError:  # deal with blank lines
                    continue
                if int(n) > 1:
                    # this is due to an int overflow bug,
                    # issue has been raised here:
                    # github.com/bcgsc/ntCard/issues/48
                    if int(n) > 2 ** 62:
                        return -1
                    return int(freq)
            return lfreq

    def calc_gldist(self):
        glen_range = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            for gene, fa_loc in self._yield_gene_cluster(tmp_dir, group=True):
                glen = []
                for rs in SeqIO.parse(fa_loc[0], 'fasta'):
                    glen.append(len(rs.seq))
                glen_range.append(max(glen) - min(glen))
        return glen_range

    @staticmethod
    def _get_slices(df, allele):
        """
        Helper function to find all the alleles in the junction.

        Parameter
        ---------
        df: pandas.Series
            Series containing the alleles, must be sorted
        allele: str
            name of allele to find

        Returns
        -------
        tuple of 2 ints
            position of the first and last appearance of allele in the Series
        """
        return df.allele.searchsorted(allele, 'left'), df.allele.searchsorted(allele, 'right')

    def _yield_genome_junction(self):
        """
        Finds all the junctions associated with the given genome. Yields data for one genome at a time.

        Parameters
        ----------
        gen: int
            number associated with the genome to search
        Yields
        -------
        tuple of arrays
           first array contains the all junctions present in the genomes, second array contains the junctions
           position within the corresponding allele.
        """
        for gen in np.arange(self.df_alleles.shape[1] - 1):
            matched_idx = np.where(self.genomes == gen)[0]
            matched_alleles_num = [self.alleles[i] for i in matched_idx]
            # print(f'SHAPE OF DF_ALLELES: {self.df_alleles.shape}')
            matched_alleles = self.df_alleles.index[matched_alleles_num]
            slices = set([self._get_slices(self.df_jct, i) for i in matched_alleles])
            if len(slices) == 0:  # no matches found for the genome
                print(f'Skipping genome {gen}, no matches found')
                return set(), set()
            matched_df = self.df_jct.iloc[np.r_.__getitem__(tuple([slice(i, j) for i, j in slices]))]
            matched_junction = np.array([self.junctions2num[i] for i in matched_df.jct.values])
            yield matched_junction, matched_df.pos.values

    def _make_memory_mtx(self, shape):
        """
        Creates a csc formatted junctions by genome sparse matrix.

        Parameters
        ----------
        shape: tuple
            tuple specifying the shape of the csc matrix
        """
        indices_list = []
        data_list = []
        indptr = np.zeros(shape[1] + 1, dtype=int)
        n_total = 0

        for doc_num, (cur_indices, cur_data) in enumerate(self._yield_genome_junction()):
            if len(cur_indices) == 0:
                continue
            n_cur = len(cur_indices)
            # n_prev = n_total
            n_total += n_cur
            indices_list.append(cur_indices)
            data_list.append(cur_data)
            indptr[doc_num + 1] = n_total
        indices = np.concatenate(indices_list)
        data = np.concatenate(data_list)
        return sparse.csc_matrix((data, indices, indptr), shape=shape)

    def _load_dfalleles(self):
        """
        Extracts all the data from df_alleles that are needed for creating junction by genome data.
        """
        # load the allele by genome file
        # with open(self.allele_file, 'rb') as f:
        #     df_alleles = pk5.load(f)
        self.df_alleles = pd.read_pickle(self.allele_file).astype('float')
        # self.df_alleles = df_alleles
        print(self.df_alleles.shape)
        # drop single alleles, i.e. gene cluster with no detected mutations
        self.df_alleles = self.df_alleles.drop(self.single_alleles)

        # num2genome = dict(zip(np.arange(df_alleles.shape[1]), df_alleles.columns))
        # allele2num = dict(zip(df_alleles.index, np.arange(df_alleles.shape[0])))

        # get alleles, genomes pairs, genomes[i] contains alleles[i]
        self.alleles, self.genomes, _ = sparse.find(sparse_arrays_to_sparse_matrix(self.df_alleles))
        self.uniq_genomes = np.unique(self.genomes)

    def _load_dfjunction(self, jct_path):
        """
        Extracts all the data from df_junctions that are needed for creating junction by genome data.
        """
        self.jct_path = jct_path
        freq_thresh = 0.01
        self.df_jct = pd.read_csv(self.jct_path, header=None, names=['junction', 'pos'])

        # drop junctions with low frequency
        self.df_jct['jct'] = self.df_jct.junction.map(lambda name: name[:name.rfind('A')] + name[name.rfind('J'):])
        freq = self.df_jct.jct.value_counts() / len(self.genomes)
        low_freq = np.where(freq.values < freq_thresh)
        self.df_jct = self.df_jct.drop(low_freq[0])  # np.where returns two arr
        self.df_jct = self.df_jct.reset_index(drop=True)

        self.df_jct['allele'] = self.df_jct.junction.apply(lambda name: name[:name.rfind('J')])
        self.df_jct.sort_values(['allele'], inplace=True)
        self.junctions2num = dict(zip(self.df_jct.jct.unique(), np.arange(len(self.df_jct))))

    def make_jct_mtx(self, jct_path, freq_thresh=0.01):
        """
        Convert output of find junctions to junctions by genome sparse matrix where the value is the position
        of the junction in the corresponding gene in the genome.

        Parameters
        ----------
        jct_path: str, path.PATH
            path to the junctions file, output on findjunction function
        freq_thresh: float
            frequency threshold to keep a junction. If junction frequency < than threshold it will
            be dropped. Must be in range [0, 1).

        Returns
        -------
        csc: scipy.sparse.csc_matrix
            csc sparse matrix with jct and position of the junction in the corresponding gene in the genome
        """

        if not 0 <= freq_thresh < 1:
            raise ValueError(f'freq_thresh must be in range [0, 1). {freq_thresh} passed instead.')

        if not self.df_alleles:
            self._load_dfalleles()

        self._load_dfjunction(jct_path)  # load junction data
        mtx_shape = (len(self.df_jct.jct.unique()), len(self.uniq_genomes))
        return self._make_memory_mtx(mtx_shape)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
