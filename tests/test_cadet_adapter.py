from pathlib import Path
import platform
import shutil
import unittest

from CADETProcess.simulator import Cadet


def detect_cadet():
    """TODO: Consider moving to Cadet module."""
    executable = 'cadet-cli'
    if platform.system() == 'Windows':
        executable += '.exe'
    cli_path = Path(shutil.which(executable))

    found_cadet = False
    if cli_path.is_file():
        found_cadet = True
    install_path = cli_path.parent.parent
    return found_cadet, cli_path, install_path


found_cadet, cli_path, install_path = detect_cadet()


class Test_Adapter(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    @unittest.skipIf(found_cadet is False, "Skip if CADET is not installed.")
    def test_install_path(self):
        simulator = Cadet()
        self.assertEqual(cli_path, simulator.cadet_cli_path)

        with self.assertWarns(UserWarning):
            simulator.install_path = cli_path
        self.assertEqual(cli_path, simulator.cadet_cli_path)

        simulator.install_path = cli_path.parent.parent
        self.assertEqual(cli_path, simulator.cadet_cli_path)

        simulator = Cadet(install_path)
        self.assertEqual(cli_path, simulator.cadet_cli_path)

        with self.assertRaises(FileNotFoundError):
            simulator = Cadet('foo/bar')

    @unittest.skipIf(found_cadet is False, "Skip if CADET is not installed.")
    def test_check_cadet(self):
        simulator = Cadet()

        self.assertTrue(simulator.check_cadet())

        file_name = simulator.get_tempfile_name()
        cwd = simulator.temp_dir
        sim = simulator.create_lwe(cwd / file_name)
        sim.run()

    @unittest.skipIf(found_cadet is False, "Skip if CADET is not installed.")
    def test_create_lwe(self):
        simulator = Cadet()

        file_name = simulator.get_tempfile_name()
        cwd = simulator.temp_dir
        sim = simulator.create_lwe(cwd / file_name)
        sim.run()

    def tearDown(self):
        shutil.rmtree('./tmp', ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
