#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 22:20:07 2021

@author: jhyun95

"""

import numpy as np
import scipy.sparse

def read_lsdf(npz_file, label_file=None):
    '''
    Creates a LightSparseDataFrame from file.
    
    Parameters
    ----------
    npz_file : str
        Path to npz file with scipy.sparse matrix
    label_file : str
        Path to text file with index and column names. 
        If None, uses <npz_file>.labels.txt (default None)
        
    Returns
    -------
    lsdf : LightSparseDataFrame
        LightSparseDataFrame constucted from files
    '''
    data = scipy.sparse.load_npz(npz_file)
    label_path = npz_file + '.labels.txt' if label_file is None else label_file
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            labels.append(line.strip())
    n_rows, n_cols = data.shape
    return LightSparseDataFrame(labels[:n_rows], labels[n_rows:], data)


class LightSparseDataFrame:
    
    def __init__(self, index, columns, data):
        '''
        Parameters
        ----------
        index : list
            List of objects to use as index labels
        columns : list
            List of objects to use as column labels
        data : scipy.sparse.spmatrix
            Table data in scipy sparse matrix format
        '''
        try:
            self.data = data.tolil()
        except: 
            print 'ERROR: Could not convert data to LIL format'
            self.data = np.nan
        self.index = index
        self.columns = columns
        self.shape = self.data.shape
        self.index_map = {index[i]:i for i in range(len(index))}
        self.column_map = {columns[i]:i for i in range(len(columns))}
        if len(index) != data.shape[0]:
            print 'ERROR: Index length does not match data'
        if len(columns) != data.shape[1]:
            print 'ERROR: Column length does no match data'
            
    def labelslice(self, indices=None, columns=None):
        '''
        Extract a dataframe slice by index and/or column labels.
        '''
        pass
        
    def islice(self, i_indices=None, i_columns=None):
        '''
        Extract a dataframe slice by index and/or column indices.
        '''
        pass
        
            
    def to_npz(self, npz_file, label_file=None):
        '''
        Save data to three files.
        
        Parameters
        ----------
        npz_file : str
            Path to output npz file with scipy.sparse matrix
        label_file : str
            Path output text file with index and column names. 
            Writes indices first, then columns, one per line.
            If None, uses <npz_file>.labels.txt (default None)
        '''
        label_path = npz_file + '.labels.txt' if label_file is None else label_file
        with open(label_path, 'w+') as f:
            for index_val in self.index:
                f.write(index_val + '\n')
            for column_val in self.columns:
                f.write(column_val + '\n')
        scipy.sparse.save_npz(npz_file, self.data.tocoo())