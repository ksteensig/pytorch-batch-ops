from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

conda = os.getenv("CUDA_HOME")
if conda:
    inc = [conda + "/include"]
else:
    inc = []

libname = "torch_batch_ops_cpp"
inc.append('/usr/local/cuda/include') 
setup(name=libname,
      ext_modules=[CppExtension(
          libname,
          ['batch_ops.cpp'],
          include_dirs=inc,
          libraries=["cusolver", "cublas"],
          extra_compile_args={'cxx': ['-g', '-DDEBUG'],
                              'nvcc': ['-O2']}
      )],
      cmdclass={'build_ext': BuildExtension})
