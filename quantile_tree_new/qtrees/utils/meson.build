py.install_sources('__init__.py', 'validate.py', subdir: 'qtrees/utils')

utils_extension_metadata = {
  '_typedefs': {'sources': ['_typedefs.pyx'], 'override_options': ['optimization=3']},
  '_random': {'sources': ['_random.pyx'], 'override_options': ['optimization=3']},
  '_utils': {'sources': ['_utils.pyx'], 'override_options': ['optimization=3']},
  '_quantile': {'sources': ['_quantile.pyx'], 'override_options': ['optimization=3']},
}

foreach ext_name, ext_dict : utils_extension_metadata
  py.extension_module(
    ext_name,
    ext_dict.get('sources'),
    dependencies: [np_dep],
    override_options: ext_dict.get('override_options', []),
    cython_args: cython_args,
    subdir: 'qtrees/utils',
    install: true
  )
endforeach
