project('qtrees', ['c', 'cpp'], version: '0.0.1')

py = import('python').find_installation()

np_inc = run_command(py, ['-c', 'import numpy; print(numpy.get_include())'], check: true).stdout().strip()
np_dep = declare_dependency(include_directories: include_directories(np_inc))

cython_args = ['--fast-fail', '-3']

subdir('qtrees')
