import os
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

def build(setup_kwargs):
    # Encontra todos os arquivos .pyx no diretório src
    extensions = []
    
    # Adicionando extensões explicitamente
    core_dir = os.path.join("src", "bz_pricing", "core")
    
    if os.path.exists(core_dir):
        for file in os.listdir(core_dir):
            if file.endswith(".pyx"):
                module_name = f"bz_pricing.core.{file[:-4]}"
                file_path = os.path.join(core_dir, file)
                extensions.append(Extension(module_name, [file_path]))
                
    if extensions:
        setup_kwargs.update({
            "ext_modules": cythonize(
                extensions,
                compiler_directives={'language_level': "3", 'boundscheck': False, 'wraparound': False},
                include_path=["src"]
            ),
            "include_dirs": [np.get_include(), "src"]
        })

if __name__ == "__main__":
    kwargs = {}
    build(kwargs)
    kwargs["package_dir"] = {"": "src"}
    setup(**kwargs)
