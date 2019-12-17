#!/usr/bin/env python
import setuptools

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE.rst') as f:
    license = f.read()


setuptools.setup(
        name='CADET-Process',
        version='0.1',
        description='Tool for modelling and optimizing chromatographic processes.',
        long_description=readme,
        author='Johannes Schmölder',
        author_email='johannes.schmoelder@fau.de',
        url='https://github.com/fau-advanced-separations/CADET-Process',
        license=license,
        packages=setuptools.find_packages(exclude=('tests', 'docs')),
        install_requires=[
                'numpy>=1.16.2',
                'scipy>=1.3.1',
                'matplotlib>=3.1.1',
                'addict>=2.2.1',
                'deap>=1.3.0',
                'GPyOpt>=1.2.5',
                'CADET>=0.2',
                ],
        classifiers=[
                "Programming Language :: Python :: 3",
                "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                "Operating System :: OS Independent",
                ],
        python_requires='>=3.6',
    )
