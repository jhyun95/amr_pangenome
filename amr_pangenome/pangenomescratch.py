

#Goal is to make function that generates the Allele table of the CDS gene seq.
#Need to get some aspects of code from build_cds_pangenome and some from
# build_non_coding pangenome, and then cluster with CD-HIT-est.

# TODO: Double check cdhit_args for cdhit-est
def build_cds_nucl_pangenome(genome_data, output_dir, name='Test', flanking=(0,0),
                             allowed_features=['CDS', 'tRNA'],
                             cdhit_args={-'n': 5}, '-c':0.8, fasta_sort_path=None,
                             save_csv=True):
    '''
        Constructs a pan-genome based on  coding nucleic acid sequences with the following steps:
        1) Extract coding transcripts based on FNA/GFF pairs
        2) Cluster CDS by sequence into putative transcripts using CD-HIT-EST
        3) Rename non-redundant transcript as <name>_T#A#, referring to transcript cluster and allele number
        4) Compile allele/transcript membership into binary transcript allele x genome and transcript x genome tables

        Generates eight files within output_dir:
        1) <name>_strain_by_coding_nuc_allele.pickle.gz, binary allele x genome table with SparseArray structure
        2) <name>_strain_by_coding_nuc.pickle.gz, binary gene x genome table with SparseArray structure
        1) <name>_strain_by_coding_nuc_allele.csv.gz, binary allele x genome table as flat file (if save_csv)
        2) <name>_strain_by_coding_nuc.csv.gz, binary gene x genome table as flat file (if save_csv)
        3) <name>_coding_nuc_nr.fna, all non-redundant coding seqs observed, with headers <name>_T#A#
        4) <name>_coding_nuc_nr.fna.cdhit.clstr, CD-HIT-EST output file from clustering
        5) <name>_coding_nuc_allele_names.tsv, mapping between <name>_T#A# to original transcript headers
        6) <name>_coding_nuc_redundant_headers.tsv, lists of headers sharing the same sequences, with the
            representative header relevant to #5 listed first for each group.
        7) <name>_coding_missing_headers.txt, lists headers for original entries missing sequences

        Parameters
        ----------
        genome_data : list
            List of 2-tuples (genome_gff, genome_fna) for use by extract_coding()
        output_dir : str
            Path to directory to generate outputs and intermediates.
        name : str
            Header to prepend to all output files and allele names (default 'Test')
        flanking : tuple
            (X,Y) where X = number of nts to include from 5' end of feature,
            and Y = number of nts to include from 3' end feature. Features
            may be truncated by contig boundaries (default (0,0))
        allowed_features : list
            List of GFF feature types to extract. Default includes
            features labeled "CDS" and "tRNA"
        cdhit_args : dict
            Alignment arguments to pass CD-HIT-EST, other than -i, -o, and -d
            (default {'-n':5, '-c':0.8})
        fastasort_path : str
            Path to Exonerate's fastasort binary, optionally for sorting
            final FAA files (default None)
        save_csv : bool
            If true, saves allele and gene tables as csv.gz. May be limiting
            step for very large tables (default True)

        Returns
        -------
        df_nuc_alleles : pd.DataFrame
            Binary non-coding allele x genome table
        df_nuc_genes : pd.DataFrame
            Binary non-coding gene x genome table
        '''

    ''' Extract coding nucleid acid sequences from all genomes '''
    print('Identiafying non-redundant CDS sequences...')
    output_nr_faa = output_dir + '/' + name '_coding_nuc_nr.faa' #final non-redundant
    output_shared_headers = output_dir + '/' + name + '_coding_nuc_redundant_headers.tsv' # records headers that have the same sequence
    output_missing_headers = output_dir + '/' + name + '_coding_nuc_missing_headers.txt' # records headers without any seqeunce
    output_nr_faa = output_nr_faa.replace('//','/')
    output_shared_headers = output_shared_headers.replace('//','/')
    output_missing_headers = output_missing_headers.replace('//','/')

    # before we consolidate the sequences, we need to generate the coding fna files
    print('Extracting coding-sequences...')
    genome_coding_paths = []
    for i, gff_fna in enumerate(genome_data):
        '''Prepare output path'''
        genome_dir = '/'.join(genome_gff.split('/')[:-1]) + '/' if '/' in genome_gff else ''
        genome_nuc_dir = genome_dir + 'derived/'  # output coding sequences here
        if not os.path.exists(genome_nc_dir):
            os.mkdir(genome_nuc_dir)
        genome_nuc = genome_nuc_dir + genome + '_nuc_coding.fna'
        '''Extract non-coding sequnces'''
        print(i+1, genome)
        genome_coding_paths.append(genome_nc)
        extract_coding_fna(genome_gff, genome_fna, genome_nc,
                          allowed_features=allowed_features)

    # consolidate redundant sequences
    non_redundant_seq_hashes, missing_headers = consolidate_seqs(
        genome_faa_paths, output_nr_faa, output_shared_headers, output_missing_headers)

    ''' Apply CD-Hit to non-redundant CDS sequences '''
    output_nr_faa_copy = output_nr_faa + '.cdhit'  # temporary FAA copy generated by CD-Hit
    output_nr_clstr = output_nr_faa + '.cdhit.clstr'  # cluster file generated by CD-Hit
    cluster_with_cdhit(output_nr_faa, output_nr_faa_copy, cdhit_args)
    os.remove(output_nr_faa_copy)  # delete CD-hit copied sequences

    ''' Extract genes and alleles, rename unique sequences as <name>_C#A# '''
    output_allele_names = output_dir + '/' + name + '_allele_names.tsv'  # allele names vs non-redundant headers
    output_allele_names = output_allele_names.replace('//', '/')
    header_to_allele = rename_genes_and_alleles(
        output_nr_clstr, output_nr_faa, output_nr_faa,
        output_allele_names, name=name, cluster_type='cds',
        shared_headers_file=output_shared_headers,
        fastasort_path=fastasort_path)
    # maps original headers to short names <name>_C#A#

    ''' Process gene/allele membership into binary tables '''
    df_alleles, df_genes = build_genetic_feature_tables(
        output_nr_clstr, genome_faa_paths, name,
        cluster_type='cds', header_to_allele=header_to_allele)

    ''' Save tables as PICKLE.GZ (preserve SparseArrays) and CSV.GZ (backup flat file) '''
    output_allele_table = output_dir + '/' + name + '_strain_by_allele'
    output_gene_table = output_dir + '/' + name + '_strain_by_gene'
    output_allele_table = output_allele_table.replace('//', '/')
    output_gene_table = output_gene_table.replace('//', '/')
    output_allele_csv = output_allele_table + '.csv.gz'
    output_gene_csv = output_gene_table + '.csv.gz'
    output_allele_pickle = output_allele_table + '.pickle.gz'
    output_gene_pickle = output_gene_table + '.pickle.gz'
    print('Saving', output_allele_pickle, '...')
    df_alleles.to_pickle(output_allele_pickle)
    print('Saving', output_gene_pickle, '...')
    df_genes.to_pickle(output_gene_pickle)
    if save_csv:
        print('Saving', output_allele_csv, '...')
        df_alleles.to_csv(output_allele_csv)
        print('Saving', output_gene_csv, '...')
        df_genes.to_csv(output_gene_csv)

    return df_alleles, df_genes


def extract_coding_fna(genome_gff, genome_fna, coding_out,
                       allowed_features=['CDS', 'tRNA']):
    '''
        Extracts nucleotides for coding sequences.
        Interprets GFFs as formatted by PATRIC:
            1) Assumes contigs are labeled "accn|<contig>".
            2) Assumes protein features have ".peg." in the ID
            3) Assumes ID = fig|<genome>.peg.#

        Parameters
        ----------
        genome_gff : str
            Path to genome GFF file with CDS coordinates
        genome_fna : str
            Path to genome FNA file with contig nucleotides
        coding_out : str
            Path to output transcript sequences FNA files
        allowed_features : list
            List of GFF feature types to extract. Default includes
            features labeled "CDS" and "tRNA"
            (default ['CDS', 'tRNA'])
    '''

    contigs = load_sequences_from_fasta(genome_fna, header_fxn=lambda x: x.split()[0])
    with open(coding_out, 'w+') as f_coding:
        with open(genome_gff, 'r') as f_gff:
            for line in f_gff:
                ''' Check for non-comment and non-empty line '''
                if not line[0] == '#' and not len(line.strip()) == 0:
                    contig, src, feature_type, start, stop, \
                    score, strand, phase, meta = line.split('\t')
                    contig = contig[5:]  # trim off "accn|" header
                    fstart = int(start)
                    fstop = int(stop)

                    if feature_type in allowed_features:
                        ''' Get noncoding feature sequence and ID '''
                        contig_seq = contigs[contig]
                        fstart = max(0, fstart)  # avoid looping due to contig boundaries
                        feature_seq = contig_seq[fstart:fstop]
                        if strand == '-':  # negative strand
                            feature_seq = reverse_complement(feature_seq)
                        meta_key_vals = [x.split('=') for x in meta.split(';')]
                        metadata = {x[0]: x[1] for x in meta_key_vals}
                        feature_id = metadata['ID']

                        ''' Save to output file '''
                        feature_seq = '\n'.join(feature_seq[i:i + 70] for i in range(0, len(feature_seq), 70))
                        f_coding.write('>' + feature_id + '\n')
                        f_coding.write(feature_seq + '\n')

