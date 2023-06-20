from pathlib import Path
import platform
import shutil
import unittest
import warnings

from CADETProcess.simulator import Cadet

executable = 'cadet-cli'
if platform.system() == 'Windows':
    executable += '.exe'
cli_path = Path(shutil.which(executable))

found_cadet = False
if cli_path.is_file():
    found_cadet = True
install_path = cli_path.parent.parent


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


if __name__ == '__main__':
    unittest.main()
