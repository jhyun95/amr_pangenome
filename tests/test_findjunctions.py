import pytest
from amr_pangenome import findjunctions
from unittest.mock import patch, mock_open




def test_makesparsematrix_readline(mockopen):
    open_mock = mock_open()
    with patch("findjunctions.make_sparse_matrix.open",
               open_mock, create=True):
        findjunctions.create_sparse_rows.write_to_file('test_data')
    open_mock.assert_called_with("output.txt", "w")
    open_mock.return_value.write.assert_called_once_with("test-data")
    # mock open graph path
    # load some sequence of data
    # check if the sequence loaded is as expected


"""
function description:
1. read the graph_path
2. split lines based on ';'
3. assign unique names to junctions
4. return a coo formatted sparse matrix
"""