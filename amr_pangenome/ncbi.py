#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 21:16:50 2021

@author: jhyun95
"""

import time, os, ftplib, shutil
import subprocess as sp

def run_prodigal_parallel(fna_paths, processes=4, poll_time=0.5,
    prodigal_args=['-c', '-m', '-g', '11', '-p', 'single', '-q'],
    prodigal_path='prodigal', footer=''):
    '''
    Runs Prodigal for multiple assemblies in parallel. For each 
    assembly file <name>.fna, creates <name><footer>.gff and 
    <name><footer>.faa.
    
    Parameters
    ----------
    fna_paths : list
        Full paths to each assembly file.
    threads : int
        Number of parallel processes (default 4)
    poll_time : float
        Seconds to wait between polling active processes (default 0.5)
    prodigal_args : list
        List of additional arguments to provide to Prodigal for all
        runs. Default uses settings from Prokka for bacteria.
        (default ['-c', '-m', '-g', '11', '-p', 'single', '-q'])
    prodigal_path : str
        Path to prodigal binary (default 'prodigal')
    footer : str
        Label to append to output file names (default '')
    '''
    active_processes = []
    remaining_fnas = list(fna_paths)
    assembly_count = 0
    while len(remaining_fnas) > 0 or len(active_processes) > 0:
        ''' Spawn Prodigal processes up to maximum specified '''
        while len(remaining_fnas) > 0 and len(active_processes) < processes:
            fna_path = remaining_fnas.pop()
            if os.path.exists(fna_path) and fna_path.endswith('.fna'):
                output_faa = fna_path[:-4] + footer + '.faa'
                output_gff = fna_path[:-4] + footer + '.gff'
                args = [prodigal_path, '-i', fna_path, '-o', output_gff, 
                        '-a', output_faa, '-f', 'gff']
                args += prodigal_args
                assembly_count += 1
                print assembly_count, args
                proc = sp.Popen(args)
                active_processes.append(proc)
        
        ''' Check processes for completed runs '''
        completed_processes = []
        for proc in active_processes:
            if not proc.poll() is None: # process completed
                proc.wait() # blocks and cleans upon process finishing?
                completed_processes.append(proc)
        for proc in completed_processes:
            active_processes.remove(proc)
        
        time.sleep(poll_time)


def download_ncbi_assemblies(accession_ids, output_dir, batch_size=100,
    datasets_prog='/home/user/notebook/jason/tools/datasets/datasets',
    ftp_url='ftp.ncbi.nlm.nih.gov'):
    '''
    Two stage approach to download NCBI assemblies.
    1) Fast batch downloads of genomes using NCBI datasets
    2) Fill missed genomes individually via NCBI FTP server
    
    See download_ncbi_assemblies_using_datasets() and
    download_ncbi_assemblies_using_ftp() for parameters.
    '''
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    download_ncbi_assemblies_using_datasets(
        accession_ids, output_dir, batch_size, datasets_prog)
    downloaded_accs = filter(lambda x: x in accession_ids, os.listdir(output_dir))
    print 'Downloaded', len(downloaded_accs), 'genomes of', len(accession_ids), 
    print 'with NCBI datasets. Downloading rest with FTP...'
    download_ncbi_assemblies_using_ftp(
        accession_ids, output_dir, ftp_url)

    
def download_ncbi_assemblies_using_datasets(
    accession_ids, output_dir, batch_size=100,
    datasets_prog='/home/user/notebook/jason/tools/datasets/datasets'):
    ''' 
    Attempts to download assembly files (.fna) from NCBI by accession ID, 
    using the NCBI datasets program. Skips files that are already downloaded.
    
    Parameters
    ----------
    accession_ids : list
        List of assembly accession IDs, i.e. GCA*, GCF*
    output_dir : str
        Path to output files. Will create a directory <acc> for
        each assembly, and output <acc>.fna in each folder
    batch_size : int
        Maximum genomes to download at a time (default 100)
    datasets_prog : str
        Path to datasets executable (default
        '/home/user/notebook/jason/tools/datasets/datasets')
    '''
    outdir = output_dir + '/' if output_dir[-1] != '/' else output_dir

    ''' Write accession IDs to temporary file '''
    target_accs, existing_accs = __filter_existing_assemblies__(accession_ids, outdir)
    print 'Downloading', len(target_accs), 'genomes',
    print '(skipping', len(existing_accs), 'genomes, already downloaded)'

    ''' Run datasets commands '''
    if len(target_accs) > 0:
        tmpdir = outdir + 'tmp/'
        if not os.path.isdir(tmpdir):
            os.mkdir(tmpdir)
            
        for batch_start in range(0, len(target_accs), batch_size):
            ''' Prepare batch to download '''
            batch_end = min(batch_start+batch_size, len(target_accs))
            batch_accs = target_accs[batch_start:batch_end]
            temp_accs_path = tmpdir + 'accs.tmp'
            with open(temp_accs_path, 'w+') as f:
                for acc in batch_accs:
                    f.write(acc + '\n')
        
            ''' Run NCBI datasets ''' 
            time_start = time.time()
            print 'Downloading genomes', batch_start+1, '...', batch_end
            args = [datasets_prog, 'download', 'genome', 'accession', '--inputfile', temp_accs_path, 
                    '--exclude-genomic-cds', '--exclude-gff3', '--exclude-protein', '--exclude-rna',
                    '--no-progressbar']
            print ' '.join(args)
            try:
                print sp.check_output(args, cwd=tmpdir)
                print 'Downloaded in', round(time.time() - time_start, 3), 'seconds'
            except:
                print 'NCBI datasets download failed'

            os.remove(temp_accs_path)
            if os.path.exists(tmpdir + 'ncbi_dataset.zip'):
                ''' Unzip payload '''
                print 'Decompressing download...'
                sp.call(['unzip', 'ncbi_dataset.zip'], cwd=tmpdir)

                ''' Move and rename genomes to target directory '''
                print 'Processing downloaded files...'
                for acc in batch_accs:
                    acc_dir = tmpdir + 'ncbi_dataset/data/' + acc + '/'
                    acc_dir_new = outdir + acc + '/'
                    if os.path.exists(acc_dir_new):
                        if len(os.listdir(acc_dir_new)) == 0:
                            os.rmdir(acc_dir_new)
                        else:
                            print 'ERROR: Output directory exists and is non-empty, skipping', acc
                            print acc_dir_new
                            print os.listdir(acc_dir_new)
                    if os.path.isdir(acc_dir) and (not os.path.exists(acc_dir_new)):
                        shutil.move(acc_dir, outdir)
                        if os.path.exists(acc_dir_new + 'sequence_report.jsonl'): # remove sequence report
                            os.remove(acc_dir_new + 'sequence_report.jsonl')
                        fna_files = filter(lambda x: x.endswith('.fna'), os.listdir(acc_dir_new))
                        fna_paths = map(lambda x: acc_dir_new + x, fna_files)
                        fna_out = acc_dir_new + acc + '.fna'
                        if len(fna_paths) > 1: # multiple FNAs, concatenate
                            with open(fna_out,'wb') as f_out:
                                for fna_path in fna_paths:
                                    with open(fna_path,'rb') as f_in:
                                        shutil.copyfileobj(f_in, f_out)
                                    os.remove(fna_path)
                        elif len(fna_paths) == 1: # single FNA, rename
                            os.rename(fna_paths[0], fna_out)
                        else: # no FNAs, print warning
                            print 'No FNAs downloaded:', acc

                ''' Remove temporary files '''
                shutil.rmtree(tmpdir)
                os.mkdir(tmpdir)

        os.rmdir(tmpdir)
    else:
        print 'Nothing to download, aborting'

        
def download_ncbi_assemblies_using_ftp(accession_ids, output_dir, ftp_url='ftp.ncbi.nlm.nih.gov'):
    '''
    Attempts to download assembly files (.fna) from NCBI by accession ID, 
    directly through the FTP server. Skips files that are already downloaded.
    Assumes gzip is available to decompress downloads.
    
    Parameters
    ----------
    accession_ids : list
        List of assembly accession IDs, i.e. GCA*, GCF*
    output_dir : str
        Path to output files. Will create a directory <acc> for
        each assembly, and output <acc>.fna in each folder
    ftp_url : str
        URL for FTP connection (default 'ftp.ncbi.nlm.nih.gov')
    '''
    outdir = output_dir + '/' if output_dir[-1] != '/' else output_dir
    target_accs, existing_accs = __filter_existing_assemblies__(accession_ids, outdir)
    
    print 'Downloading', len(target_accs), 'genomes',
    print '(skipping', len(existing_accs), 'genomes, already downloaded)'
    if len(target_accs) > 0:
        ftp = ftplib.FTP(ftp_url)
        ftp.login()
        for a, acc in enumerate(target_accs):
            acc_dir = outdir + acc + '/'
            fna_out = acc_dir + acc + '.fna.gz'
            fna_out_final = acc_dir + acc + '.fna'
            if not os.path.exists(acc_dir):
                os.mkdir(acc_dir)
            sub_path = '/'.join([acc[:3], acc[4:7], acc[7:10], acc[10:13]])
            sub_path = '/genomes/all/' + sub_path + '/'

            ''' Identify specific assembly of interest '''
            try:
                ftp.cwd(sub_path)
                assembly = filter(lambda x: x.startswith(acc), ftp.nlst())
            except: # first fail
                print '\tWARN: Identifying assembly failed, retrying for', acc
                try: 
                    time.sleep(1.0)
                    ftp.cwd(sub_path)
                    assembly = filter(lambda x: x.startswith(acc), ftp.nlst())
                except: # second fail
                    print '\tERROR: Failed to identify assembly twice, skipping', acc
                    assembly = []

            ''' Download the specific assembly '''
            if len(assembly) == 1:
                assembly_name = assembly[0]
                fna_path = assembly_name + '/' + assembly_name + '_genomic.fna.gz'
                print a+1, acc, '\n', ftp_url + sub_path + fna_path
                try:
                    ftp.retrbinary('RETR ' + fna_path, open(fna_out, 'w+').write)
                    sp.call(['gzip', '-d', fna_out])
                except:
                    print '\tERROR: Failed to download for', acc
                    for f in [fna_out, fna_out_final]: # clean up potentially failed downloads
                        if os.path.exists(f):
                            os.remove(f)
            elif len(assembly) == 0:
                print a+1, 'WARN: No assemblies found, skipping:',  acc
            elif len(assembly) > 1:
                print a+1, 'WARN: Multiple assemblies found, skipping:', acc
        ftp.close()

    
def __filter_existing_assemblies__(accession_ids, outdir):
    ''' Separates accession IDs into ones missing/already downloaded  '''
    target_accs = []; existing_accs = []
    for acc in accession_ids:
        acc_dir = outdir + acc + '/'
        fna_out_final = acc_dir + acc + '.fna'
        if not os.path.exists(fna_out_final): 
            target_accs.append(acc)
        else: # target file already exists
            existing_accs.append(acc)
    return target_accs, existing_accs
