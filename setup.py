"""Optional exact CPU geometry extension; project metadata stays in pyproject."""
import hashlib
import json
from pathlib import Path

from setuptools import Extension, setup

source = Path(__file__).parent / 'models/csrc/fps_cpu.c'
grid_source = Path(__file__).parent / 'models/csrc/spatial_index_cpu.cpp'
flags = ['-O3', '-fno-fast-math', '-ffp-contract=off']
grid_flags = flags + ['-pthread']
grid_link_flags = ['-pthread']
setup(ext_modules=[Extension('models._fps_cpu', sources=['models/csrc/fps_cpu.c'],
    extra_compile_args=flags, optional=True,
    define_macros=[('FPS_SOURCE_SHA256', json.dumps(hashlib.sha256(source.read_bytes()).hexdigest())),
                   ('FPS_COMPILE_FLAGS', json.dumps(' '.join(flags)))]),
    Extension('models._spatial_index_cpu', sources=['models/csrc/spatial_index_cpu.cpp'],
        extra_compile_args=grid_flags, extra_link_args=grid_link_flags,
        optional=True, language='c++',
        define_macros=[('GRID_SOURCE_SHA256', json.dumps(hashlib.sha256(grid_source.read_bytes()).hexdigest())),
                       ('GRID_COMPILE_FLAGS', json.dumps(' '.join(grid_flags))),
                       ('GRID_LINK_FLAGS', json.dumps(' '.join(grid_link_flags)))])])
