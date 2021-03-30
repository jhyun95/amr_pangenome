import pytest
from amr_pangenome import findjunctions
from unittest import mock
import os
import sys
# sys.path.append('../amr_pangenome/')


init_org_arg1, init_org_expected1 = ('CC8', '/path/to/file'), 'CC8'
findjunctions_init_params = [(init_org_arg1, init_org_expected1)]
res_dir_target = 'amr_pangenome.findjunctions.FindJunctions.res_dir'
fa_file_target = 'amr_pangenome.findjunctions.FindJunctions.fa_file'
single_allele_target = 'amr_pangenome.findjunctions.FindJunctions.get_single_alleles'

@pytest.mark.parametrize("arg, expected", findjunctions_init_params)
@mock.patch(single_allele_target)
@mock.patch(fa_file_target)
@mock.patch(res_dir_target)
def test_findjunctions_init_org(mock_isdir, mock_isfa, mock_single_allele,
                                arg, expected):
    mock_isdir.return_value = arg[1]
    mock_isfa.return_value = os.path.join(arg[1], arg[0] + '.fa')
    mock_single_allele.return_value = ['allele']
    fj = findjunctions.FindJunctions(*arg)
    assert fj.org == expected








"next we have to find a way to mock the isdir check and and the isfile check"




# makesparsematrix_readline_params = [('output.txt', 'asdfasdf')]
# @pytest.mark.parametrize('_input, expected', makesparsematrix_readline_params)
# @patch('amr_pangenome.findjunctions.FindJunctions.get_junction_data.open',
#        new_callable=mock_open(), create=True)
# def test_makesparsematrix_readline(open_mock, _input, expected):
#     open_mock.read_data = _input
#     res = findjunctions.FindJunctions.get_junction_data('output.txt')
#     assert res == expected


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