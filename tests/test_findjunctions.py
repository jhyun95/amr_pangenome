from amr_pangenome import findjunctions, ROOT_DIR
import pytest
from unittest import mock
import os
import pandas as pd
from scipy import sparse

init_args = ('CC8', '/path/to/file/Test.fna')
fa_file_target = 'amr_pangenome.findjunctions.FindJunctions.fna_file'
single_allele_target = 'amr_pangenome.findjunctions.FindJunctions.get_single_alleles'


@pytest.fixture(scope='function')
def mock_findjunction():
    mock_findjunction = mock.Mock(findjunctions.FindJunctions,
                                  wraps=findjunctions.FindJunctions)
    mock_findjunction.fna_file = '/dev/null/Test.fna'
    mock_findjunction.org = 'Test'
    mock_findjunction.single_alleles = []
    return mock_findjunction


@pytest.fixture(scope='function')
def fj_single_cluster():
    fna_file = 'tests/test_data/test_single_gene_cluster_fasta.fna'
    fj = findjunctions.FindJunctions('Test', os.path.join(ROOT_DIR, fna_file))
    return fj


@pytest.fixture(scope='function')
def fj_multi_cluster():
    fna_file = 'tests/test_data/test_multi_gene_cluster_fasta.fna'
    fj = findjunctions.FindJunctions('Test', os.path.join(ROOT_DIR, fna_file))
    return fj


@pytest.fixture(scope='module')
def junctions():
    return ['A0J0', 'A1J0', 'A2J0', 'A1J1', 'A2J1']


@pytest.fixture
def positions():
    return [0, 0, 3, 5, 5, 29, 32]


@pytest.mark.parametrize("args, expected", [(init_args, 'CC8')])
@mock.patch('amr_pangenome.findjunctions.SeqIO.parse')
@mock.patch('amr_pangenome.findjunctions.FindJunctions.get_single_alleles')
@mock.patch('amr_pangenome.findjunctions.FindJunctions.fna_file')
def test_findjunctions_init_org(mock_isfa, mock_single_allele, mock_seqio,
                                args, expected):
    mock_isfa.return_value = os.path.join(args[1], args[0] + '.fa')
    mock_single_allele.return_value = ['allele']
    mock_seqio.return_value = ''
    fj = findjunctions.FindJunctions(*args)
    assert fj.org == expected


expected_fa = os.path.join(init_args[1], init_args[0] + '.fa')


# test that the proper exceptions are raised
@pytest.mark.parametrize("args, expected", [(init_args, expected_fa)])
@mock.patch(single_allele_target)
def test_findjunctions_init_fasta_exception(mock_single_allele,
                                            args, expected):
    mock_single_allele.return_value = ['allele']
    with pytest.raises(FileNotFoundError):
        findjunctions.FindJunctions(*args)


def test_findjunctions_init_single_alleles(mock_findjunction):
    # patch the imported pandas function to overwrite them
    names = ['Test_C11A1', 'Test_C11A2', 'Test_C21A0']
    findjunctions.FindJunctions.get_single_alleles(mock_findjunction, names)
    assert mock_findjunction.single_alleles == ['Test_C21A0']


@mock.patch('amr_pangenome.findjunctions.os.path.isdir')
def test_calc_junctions_outfmt_valuerror(os_path_isdir, mock_findjunction):
    os_path_isdir.return_value = True
    with pytest.raises(ValueError):
        findjunctions.FindJunctions.calc_junctions(mock_findjunction, outfmt='others')


@mock.patch('amr_pangenome.findjunctions.os.path.isdir')
@mock.patch('amr_pangenome.findjunctions.os.path.join')
def test_calc_junctions_outfmt_notimplemented(os_path_join, os_path_isdir, mock_findjunction):
    os_path_isdir.return_value = True
    os_path_join.return_value = ''
    with pytest.raises(NotImplementedError):
        findjunctions.FindJunctions.calc_junctions(mock_findjunction, outfmt='gfa2')


two_paco_dir = os.path.join(ROOT_DIR, 'bin/twopaco')


@mock.patch('amr_pangenome.findjunctions.os.path.join')
@mock.patch('amr_pangenome.findjunctions.tempfile.TemporaryDirectory')
def test_run_twopaco_process_error(patch_tmpdir, os_path_join):
    os_path_join.return_value = two_paco_dir
    patch_tmpdir.__enter__.return_value.name = '/dev/null/'
    patch_tmpdir.__str__.return_value = '/dev/null/'
    # should fail since 'fail' is passed instead of an int or str(int)
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        # noinspection PyTypeChecker
        findjunctions.FindJunctions.run_twopaco(['/dev/null/'], 'fail', 36, 'tempdir')
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 1


@mock.patch('amr_pangenome.findjunctions.subprocess.check_output')
def test_run_twopaco_cmd_line(subprocess_checkoutput):
    subprocess_checkoutput.return_value = 0
    db_out = findjunctions.FindJunctions.run_twopaco(['/dev/null'], 25, 36, 'tempdir')
    assert db_out == 'tempdir/debrujin.bin'


def test_run_graphdump_cmd_line():
    db_out = os.path.join(ROOT_DIR, 'tests/test_data/test_single_gene_cluster_debrujin.bin')
    output = findjunctions.FindJunctions.run_graphdump(db_out, 5, 'group')
    assert output == '0 0; \n0 4; 1 4; \n0 7; 1 7; \n1 0; 2 0; \n1 2; 2 2; \n2 7; \n'


def test_get_junction_data():
    fa_list = ['fa0.fna', 'fa1.fna', 'fa2.fna']
    input_file = os.path.join(ROOT_DIR, 'tests/test_data/test_graphdump_output.txt')
    with open(input_file, 'r') as graphin:
        data = '\n'.join(graphin.readlines())
        junction_data, pos_data = findjunctions.FindJunctions.get_junction_data(data, fa_list)
        assert pos_data == ['31', '31', '31', '50', '50']
        assert junction_data == ['fa0J0', 'fa1J0', 'fa2J0', 'fa1J1', 'fa2J1']


@mock.patch('builtins.open', new_callable=mock.mock_open)
def test_write_jct_file(mock_open, junctions, positions):
    findjunctions.FindJunctions.write_jct_file(junctions, positions, '/dev/null/text.txt')
    mock_open.assert_called_with('/dev/null/text.txt', 'a+')

    handle = mock_open()
    mocked_calls = [mock.call('A0J0,0\n'), mock.call('A1J0,0\n'), mock.call('A2J0,3\n'),
                    mock.call('A1J1,5\n'), mock.call('A2J1,5\n')]
    handle.write.assert_has_calls(mocked_calls)


def test_get_junction_data_file_exits_error(mock_findjunction, tmp_path):
    name = str(mock_findjunction.org) + '_jct.csv'
    outname = tmp_path / name
    outname.write_text('test')

    with pytest.raises(FileExistsError):
        findjunctions.FindJunctions.calc_junctions(mock_findjunction, outname=outname)


def test_get_junction_data_outfmt_value_error(mock_findjunction):
    with pytest.raises(ValueError):
        findjunctions.FindJunctions.calc_junctions(mock_findjunction, outfmt='gfa1')


def test_get_junction_data_even_kmer_value_error(mock_findjunction):
    with pytest.raises(ValueError):
        findjunctions.FindJunctions.calc_junctions(mock_findjunction, kmer=2)


@pytest.mark.slow
@mock.patch('amr_pangenome.findjunctions.tempfile.TemporaryDirectory')
def test_get_junction_data_single_cluster(redirect_temp, tmp_path, fj_single_cluster):

    redirect_temp.return_value = tmp_path
    out_path = os.path.join(tmp_path, fj_single_cluster.org + '_jct.csv')
    fj_single_cluster.calc_junctions(kmer=5, outname=out_path)
    # check the coo text output
    expected_jct = os.path.join(ROOT_DIR, 'tests/test_data/test_single_gene_cluster_jct.txt')
    with open(expected_jct, 'r') as expect:
        with open(out_path, 'r') as output:
            assert sorted(expect.readlines()) == sorted(output.readlines())


jct_dir = os.path.join(ROOT_DIR, 'tests/test_data/test_multi_gene_cluster_jct.txt')


@pytest.mark.slow
@pytest.mark.parametrize('args, outdir', [({'max_processes': 1}, jct_dir),
                                          ({'max_processes': 4}, jct_dir)])
def test_get_junction_data_multi_cluster(args, outdir, tmp_path, fj_multi_cluster):
    """ Test the full run on fasta with multiple clusters with 1 and 4 cores"""
    out_path = os.path.join(tmp_path, fj_multi_cluster.org + '_jct.csv')
    fj_multi_cluster.calc_junctions(outname=out_path, kmer=5,  **args)

    # don't check graphdump since thats overwritten with each gene cluster
    with open(outdir, 'r') as expect:
        with open(out_path, 'r') as output:
            assert sorted(expect.readlines()) == sorted(output.readlines())


def test_make_strain_junction_df_direct_error():
    with pytest.raises(NotADirectoryError):
        findjunctions.FindJunctions.make_junction_strain_df('jct_file', 'mock_df',
                                                            outfile='/dev/null/junctions_df.pickle.gz')


def test_make_junction_strain_df(tmp_path):
    jct_file = os.path.join(ROOT_DIR, 'tests/test_data/test_single_gene_cluster_jct.txt')
    df_files = os.path.join(ROOT_DIR, 'tests/test_data/test_single_gene_cluster_allele_genome.pickle.gz')
    expected_file = os.path.join(ROOT_DIR, 'tests/test_data/test_single_gene_cluster_junction_df.pickle.gz')
    df_alleles = pd.read_pickle(df_files)
    outfile = str(tmp_path / 'junction_df.pickle.gz')

    test_output = findjunctions.FindJunctions.make_junction_strain_df(jct_file, df_alleles, savefile=True,
                                                                      outfile=outfile)
    expected = pd.read_pickle(expected_file)
    assert all(expected.eq(test_output))


ntcard_dir = os.path.join(ROOT_DIR, 'bin/ntcard')


@pytest.mark.skip(reason='need to implement mock junction')
@mock.patch('amr_pangenome.findjunctions.os.path.join')
def test_run_nt_card_process_error(os_path_join, mock_findjunction):
    os_path_join.return_value = ntcard_dir
    # should fail since 'fail' is passed instead of an int or str(int)
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        # noinspection PyTypeChecker
        findjunctions.FindJunctions._run_ntcard(mock_findjunction,
                                                '/file/doesnt/exist.fna',
                                                '/dev/null/out.txt')
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 1


multi_clster_fa_file = os.path.join(ROOT_DIR, 'tests/test_data/test_multi_gene_cluster_fasta.fna')


@mock.patch('amr_pangenome.findjunctions.os.path.join')
def test_run_ntcard(os_path_join, mock_findjunction, tmp_path):
    os_path_join.return_value = ntcard_dir
    outfile = tmp_path / 'out.txt'
    freq = findjunctions.FindJunctions._run_ntcard(mock_findjunction, multi_clster_fa_file, outfile,
                                                   kmers=(5,))
    assert freq == 13


def test_calc_max():
    filein = os.path.join(ROOT_DIR, 'tests/test_data/test_calc_max_data.txt')
    assert findjunctions.FindJunctions._calc_max(filein) == 41


def test_calc_kmer(fj_multi_cluster):
    assert fj_multi_cluster.calc_kmer(kmers=(5,)) == [13, 15]


def test_calc_gldist(fj_multi_cluster):
    assert fj_multi_cluster.calc_gldist() == [1, 0]


