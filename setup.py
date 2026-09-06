# -*- coding: utf-8 -*-
from setuptools import setup

package_dir = \
{'': 'src'}

packages = \
['bz_pricing', 'bz_pricing.core']

package_data = \
{'': ['*']}

install_requires = \
['google-genai>=2.22.0,<3.0.0',
 'numpy>=1.24.0,<2.0.0',
 'pymupdf>=1.28.2,<2.0.0']

setup_kwargs = {
    'name': 'bz_pricing',
    'version': '0.0.1',
    'description': 'Biblioteca (Cython/Python) para precificação Onshore e Offshore.',
    'long_description': '# bz_pricing\n\nBiblioteca Quantitativa para precificação de Renda Fixa e Derivativos (Onshore e Offshore) baseada em Cython.\n',
    'author': 'Guilherme Azevedo [tolotti.azevedo@gmail.com]',
    'author_email': 'None',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'package_dir': package_dir,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.10,<4.0',
}
from build import *
build(setup_kwargs)

setup(**setup_kwargs)
