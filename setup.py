"""Optional exact CPU geometry extension; project metadata stays in pyproject."""
import hashlib
import json
from pathlib import Path

from setuptools import Extension, setup

source = Path(__file__).parent / 'models/csrc/fps_cpu.c'
flags = ['-O3', '-fno-fast-math', '-ffp-contract=off']
setup(ext_modules=[Extension('models._fps_cpu', sources=['models/csrc/fps_cpu.c'],
    extra_compile_args=flags, optional=True,
    define_macros=[('FPS_SOURCE_SHA256', json.dumps(hashlib.sha256(source.read_bytes()).hexdigest())),
                   ('FPS_COMPILE_FLAGS', json.dumps(' '.join(flags)))])])
