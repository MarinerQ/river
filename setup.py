import numpy
from setuptools import setup


install_requires = [
    'numpy',
    'bilby',
]

setup(
    name='river',
    version="0.0.1",
    description='Gravitational waves data process with machine learning',
    author='Qian Hu',
    author_email='q.hu.2@research.gla.ac.uk',
    url='https://github.com/marinerq/river',
    license='MIT',
    python_requires='>=3',
    packages=['river', 'river.data', 'river.models'],
    install_requires=install_requires,
    include_dirs=[numpy.get_include()],
    setup_requires=['numpy'],
    entry_points={},
)
