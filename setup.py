"""Optional exact CPU geometry extension; project metadata stays in pyproject."""
import json

from setuptools import Extension, setup

flags = ['-O3', '-fno-fast-math', '-ffp-contract=off']
grid_flags = flags + ['-pthread']
grid_link_flags = ['-pthread']
setup(ext_modules=[Extension('models._fps_cpu', sources=['models/csrc/fps_cpu.c'],
    extra_compile_args=flags, optional=True,
    define_macros=[('FPS_SOURCE_VERSION', json.dumps('fps_cpu_v1')),
                   ('FPS_COMPILE_FLAGS', json.dumps(' '.join(flags)))]),
    Extension('models._spatial_index_cpu', sources=['models/csrc/spatial_index_cpu.cpp'],
        extra_compile_args=grid_flags, extra_link_args=grid_link_flags,
        optional=True, language='c++',
        define_macros=[('GRID_SOURCE_VERSION', json.dumps('spatial_index_cpu_v1')),
                       ('GRID_COMPILE_FLAGS', json.dumps(' '.join(grid_flags))),
                       ('GRID_LINK_FLAGS', json.dumps(' '.join(grid_link_flags)))])])
