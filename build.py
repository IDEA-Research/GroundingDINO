import os
import glob 

from torch.utils.cpp_extension import CppExtension

this_dir = os.path.dirname(os.path.abspath(__file__))
extensions_dir = os.path.join(this_dir, "groundingdino", "models", "GroundingDINO", "csrc")
main_source = os.path.join(extensions_dir, "vision.cpp")
base_sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"))
sources = [main_source] + [os.path.join(extensions_dir, s) for s in base_sources]
include_dirs = [extensions_dir]
define_macros = [("WITH_HIP", None)]
extra_compile_args = {"cxx": []}

custom_extension = CppExtension(
  "groundingdino._C",
  sources,
  include_dirs=include_dirs,
  define_macros=define_macros,
  extra_compile_args=extra_compile_args
)

def build(setup_kwargs):
  setup_kwargs.update(
    {"ext_modules": [custom_extension]}
  )
