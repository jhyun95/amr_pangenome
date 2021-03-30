import pytest
from amr_pangenome import findjunctions
from unittest.mock import patch, mock_open
import sys
sys.path.append('../amr_pangenome/')


arg1, expected1 = ('CC8', '/home/saugat/Documents/CC8_fasta/CDhit_res'), 'CC8'
findjunctions_init_params = [(arg1, expected1)]


@pytest.mark.parametrize("arg, expected", findjunctions_init_params)
def test_findjunctions_init(arg, expected):
    fj = findjunctions.FindJunctions(*arg)
    assert fj.org == expected









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