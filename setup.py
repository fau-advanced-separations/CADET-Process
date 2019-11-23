#!/usr/bin/env python
import setuptools

setuptools.setup(
        name='CADET-Process',
        version='0.1',
        description='Tool for modelling and optimizing advanced chromatographic processes.',
        author='Johannes Schmölder',
        author_email='johannes.schmoelder@fau.de',
        packages=setuptools.find_packages(),
        data_files=[('config', ['settings.json'])],
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
