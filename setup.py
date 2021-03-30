#!/usr/bin/env python3

import setuptools

# TODO: add install_requires=open('requirements.txt').read()
setuptools.setup(
    name='AMR pangenome',
    version='0.1.0',
    authors='Jason Huynh, Saugat Poudel',
    description='Python package for pangenome amr analysis',
    maintainer='Saugat Poudel',
    url='https://github.com/jhyun95/amr_pangenome',
    package=setuptools.find_namespace_packages(),
    python_requires='>3.7.9',
    include_packages=True,
    platforms='GNU/Linux'
)
