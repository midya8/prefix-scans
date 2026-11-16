"""Check every test and complete command against an installed wheel outside checkout.

Run with a Python environment containing setuptools and wheel. Temporary build
and install directories are removed on exit. No runtime dependency is needed.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ENV = os.environ.copy()
ENV.pop('PYTHONPATH', None)
COMMANDS = [('demo',), ('audit', 'examples/input.json'), ('adjoint-demo',),
            ('adjoint', 'examples/adjoint-input.json'), ('dense-demo',),
            ('dense', 'examples/dense-input.json'), ('curvature', 'examples/curvature-input.json')]



def run(args, cwd):
    return subprocess.check_output(list(map(str, args)), cwd=cwd, env=ENV)


def main():
    expected = [run([sys.executable, '-m', 'scanlab', *args], ROOT) for args in COMMANDS]
    assert expected[0] == (ROOT / 'examples/demo.json').read_bytes()
    assert expected[4] == (ROOT / 'examples/dense-demo.json').read_bytes()
    assert expected[2] == (ROOT / 'examples/adjoint-demo.json').read_bytes()
    with tempfile.TemporaryDirectory(prefix='scanlab-wheel-') as directory:
        temporary = Path(directory)
        run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
             '--wheel-dir', temporary / 'dist', ROOT], temporary)
        run([sys.executable, '-m', 'venv', temporary / 'env'], temporary)
        python = temporary / 'env' / 'bin' / 'python'
        wheels = list((temporary / 'dist').glob('*.whl'))
        assert len(wheels) == 1
        run([python, '-m', 'pip', 'install', '--no-deps', wheels[0]], temporary)
        outside = temporary / 'outside'
        outside.mkdir()
        for name in ('examples', 'assets', 'tests'):
            shutil.copytree(ROOT / name, outside / name, ignore=shutil.ignore_patterns('__pycache__'))
        check = '''import importlib, pathlib, sys, unittest
names = ['scanlab', 'scanlab.exact', 'scanlab.demo', 'scanlab.floating',
         'scanlab.gated', 'scanlab.lti', 'scanlab.adjoint', 'scanlab.dense', 'scanlab.curvature', 'scanlab.__main__']
modules = [importlib.import_module(name) for name in names]
def origins():
    for module in modules:
        assert pathlib.Path(module.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve()), module.__file__
origins()
sys.path.insert(0, str(pathlib.Path.cwd()))
suite = unittest.defaultTestLoader.discover('tests')
result = unittest.TextTestRunner(verbosity=2).run(suite)
origins()
assert result.wasSuccessful()
print('Full installed suite passed; all ten package origins checked before and after')
'''
        print(run([python, '-I', '-c', check], outside).decode(), end='')
        for args, original in zip(COMMANDS, expected):
            assert run([python, '-I', '-m', 'scanlab', *args], outside) == original, args
        for renderer in ('render_demo.py', 'render_adjoint.py', 'render_dense.py'):
            run([python, '-I', outside / 'examples' / renderer, '--check'], outside)
        print('All seven complete module CLI outputs match; all three original figures regenerate')


if __name__ == '__main__':
    main()
