from setuptools import find_packages, setup

setup(
    name='mlnogaster',
    packages=find_packages(),
    version='0.1.0',
    description='machine learning network of genetic algorithms simulating topologically evolved representations',
    author='michaelmech',
    install_requires=['gplearn','tpot','dask_ml','distributed','pygad','sklego']
)


