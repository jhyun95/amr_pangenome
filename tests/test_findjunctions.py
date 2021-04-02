from amr_pangenome import findjunctions, ROOT_DIR
import pytest
from unittest import mock
import os
import pandas as pd
from scipy import sparse


init_args = ('CC8', '/path/to/file')
res_dir_target = 'amr_pangenome.findjunctions.FindJunctions.res_dir'
fa_file_target = 'amr_pangenome.findjunctions.FindJunctions.fa_file'
single_allele_target = 'amr_pangenome.findjunctions.FindJunctions.get_single_alleles'


@pytest.fixture(scope='function')
def mock_findjunction():
    mock_findjunction = mock.Mock(findjunctions.FindJunctions)
    mock_findjunction.res_dir = '/dev/null'
    mock_findjunction.fa_file = 'Test.fna'
    mock_findjunction.org = 'Test'
    return mock_findjunction


@pytest.fixture(scope='module')
def junctions():
    return ['A0J0', 'A1J0', 'A2J0', 'A1J1', 'A2J1']


@pytest.fixture
def positions():
    return [0, 0, 3, 5, 5, 29, 32]


@pytest.mark.parametrize("args, expected", [(init_args, 'CC8')])
@mock.patch('amr_pangenome.findjunctions.pd.read_pickle')
@mock.patch(single_allele_target)
@mock.patch(fa_file_target)
@mock.patch(res_dir_target)
def test_findjunctions_init_org(mock_isdir, mock_isfa, mock_single_allele, mock_read_pickle,
                                args, expected):
    mock_isdir.return_value = args[1]
    mock_isfa.return_value = os.path.join(args[1], args[0] + '.fa')
    mock_single_allele.return_value = ['allele']
    mock_read_pickle.return_value = ''
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


@mock.patch('amr_pangenome.findjunctions.pd.read_pickle')
def test_findjunctions_init_single_alleles(mock_read_pickle, mock_findjunction):
    # patch the imported pandas function to overwrite them
    mock_read_pickle.return_value = ''
    df_alleles = pd.DataFrame(index=['Test_C11A1', 'Test_C11A2', 'Test_C21A0'])
    findjunctions.FindJunctions.get_single_alleles(mock_findjunction, df_alleles)
    assert mock_findjunction.single_alleles == ['Test_C21A0']


@mock.patch('amr_pangenome.findjunctions.os.path.isdir')
def test_calc_junctions_outfmt_valuerror(os_path_isdir, mock_findjunction):
    os_path_isdir.return_value = True
    with pytest.raises(ValueError):
        findjunctions.FindJunctions.calc_junctions(mock_findjunction, outfmt='others')


@mock.patch('amr_pangenome.findjunctions.os.path.isdir')
def test_calc_junctions_outfmt_notimplemented(os_path_isdir):
    os_path_isdir.return_value = True
    mock_findjunction = mock.Mock(findjunctions.FindJunctions)
    with pytest.raises(NotImplementedError):
        findjunctions.FindJunctions.calc_junctions(mock_findjunction, outfmt='gfa2')


two_paco_dir = os.path.join(ROOT_DIR, 'bin/twopaco')


@mock.patch('amr_pangenome.findjunctions.os.path.join')
@mock.patch('amr_pangenome.findjunctions.os.listdir')
def test_run_twopaco_process_error(os_path_listdir, os_path_join):
    os_path_listdir.return_value = ['fa1', 'fa2']
    os_path_join.return_value = two_paco_dir  # path to compiled two_paco
    # should fail since 'fail' is passed instead of an int or str(int)
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        # noinspection PyTypeChecker
        findjunctions.FindJunctions.run_twopaco(['/dev/null/'], 'fail', 'tempdir')
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 1


@mock.patch('amr_pangenome.findjunctions.subprocess.check_output')
def test_run_twopaco_cmd_line(subprocess_checkoutput):
    subprocess_checkoutput.return_value = 0
    db_out = findjunctions.FindJunctions.run_twopaco(['/dev/null'], 35, 'tempdir')
    assert db_out == 'tempdir/debrujin.bin'


@mock.patch('amr_pangenome.findjunctions.subprocess.call')
@mock.patch('builtins.open', new_callable=mock.mock_open)
def test_run_graphdump_cmd_line(mock_open, subprocess_call):
    subprocess_call.return_value = 0
    mock_open.read_data = ''
    graph_path = findjunctions.FindJunctions.run_graphdump('db_out', 35, 'outfmt',
                                                           'outdir')
    assert graph_path == 'outdir/graphdump.txt'


def test_get_junction_data():
    fa_list = ['fa0.fna', 'fa1.fna', 'fa2.fna']
    output = os.path.join(ROOT_DIR, 'tests/test_data/test_graphdump_output.txt')
    junction_data, pos_data = findjunctions.FindJunctions.get_junction_data(output, fa_list)
    assert pos_data == ['31', '31', '31', '50', '50']
    assert junction_data == ['fa0J0', 'fa1J0', 'fa2J0', 'fa1J1', 'fa2J1']


# TODO: DO THE THINGS BELOW FOR INTEGRATION TEST OF THE NEW FUNCTIONS
# TODO: Its going to too tough to patch and mock fasta file. so just mock the class
# TODO: fa_file and point to a test file.
# TODO: set up the test fa_generator with the test fasta files for group_seq too
@pytest.mark.skip
def test_make_strain_junction_df(mock_findjunction):
    array = pd.array([[0, 1, 0], [1, 0, 1], [1, 1, 1]])
    df_alleles = pd.DataFrame(array, index=['A0', 'A1', 'A2'],
                              columns=['fasta1', 'fasta2', 'fasta3'])
    mock_findjunction.junction_row_idx = junctions
    mock_findjunction.pos_data = positions
    res = findjunctions.FindJunctions.make_junction_strain_df(mock_findjunction, df_alleles)
    assert res.dtype == int
    assert res.index == junctions
    assert res.columns == df_alleles.columns


@mock.patch('builtins.open', new_callable=mock.mock_open)
def test_write_coo_file(mock_open, junctions, positions):
    findjunctions.FindJunctions.write_coo_file(junctions, positions, '/dev/null/text.txt')
    mock_open.assert_called_with('/dev/null/text.txt', 'a+')

    handle = mock_open()
    mocked_calls = [mock.call('A0J0,0\n'), mock.call('A1J0,0\n'), mock.call('A2J0,3\n'),
                    mock.call('A1J1,5\n'), mock.call('A2J1,5\n')]
    handle.write.assert_has_calls(mocked_calls)


"""
creates a sparse junction by fasta matrix from junction and position data.
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
 
 should throw error if not match for the fasta is found
"""
