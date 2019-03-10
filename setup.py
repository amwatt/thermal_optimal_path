import setuptools


with open('requirements.txt') as f:
    requirements = f.read().splitlines()


with open('README.rst') as f:
    long_description = f.read()


setuptools.setup(
    name='thermal_optimal_path',
    version='0.1',
    install_requires=requirements,
    long_description=long_description,
    packages=['thermal_optimal_path']
)
