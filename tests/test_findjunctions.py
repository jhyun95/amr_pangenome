from amr_pangenome import findjunctions
import pytest
from unittest import mock
import os
import pandas

init_args = ('CC8', '/path/to/file')
res_dir_target = 'amr_pangenome.findjunctions.FindJunctions.res_dir'
fa_file_target = 'amr_pangenome.findjunctions.FindJunctions.fa_file'
single_allele_target = 'amr_pangenome.findjunctions.FindJunctions.get_single_alleles'


@pytest.mark.parametrize("args, expected", [(init_args, 'CC8')])
@mock.patch(single_allele_target)
@mock.patch(fa_file_target)
@mock.patch(res_dir_target)
def test_findjunctions_init_org(mock_isdir, mock_isfa, mock_single_allele,
                                args, expected):
    mock_isdir.return_value = args[1]
    mock_isfa.return_value = os.path.join(args[1], args[0] + '.fa')
    mock_single_allele.return_value = ['allele']
    fj = findjunctions.FindJunctions(*args)
    assert fj.org == expected


expected_fa = os.path.join(init_args[1], init_args[0] + '.fa')


# test that the proper exceptions are raised
@pytest.mark.parametrize("args, expected", [(init_args, expected_fa)])
@mock.patch(single_allele_target)
@mock.patch(res_dir_target)
def test_findjunctions_init_fasta_exception(mock_isdir, mock_single_allele,
                                            args, expected):
    mock_isdir.return_value = True
    mock_single_allele.return_value = ['allele']
    with pytest.raises(FileNotFoundError):
        findjunctions.FindJunctions(*args)


@pytest.mark.parametrize("args, expected", [(init_args, expected_fa)])
@mock.patch(single_allele_target)
@mock.patch(fa_file_target)
def test_findjunctions_init_resdir_exception(mock_isfa, mock_single_allele,
                                             args, expected):
    mock_isfa.return_value = True
    mock_single_allele.return_value = ['allele']
    with pytest.raises(NotADirectoryError):
        findjunctions.FindJunctions(*args)

#make fake df_alleles and pass that with mock

# @mock.patch(fa_file_target)
# def test_findjunctions_init_single_alleles():



"""
function description:
1. read the graph_path
2. split lines based on ';'
3. assign unique names to junctions
4. return a coo formatted sparse matrix
"""

"""
COO format 
(data, (i, j)) where data, i and j are iterables containing data, row index and
column index respectively.

data is the position in the genome, what to do with sentinel junction with 0 positions?
row index is the unique name for the junction
column index is the name for the fasta file


we don't have to determine the name and the column index at first,
 the create_sparse_rows should be 
"""