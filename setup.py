from distutils.core import setup
from setuptools import find_packages
setup(
    name='oolearning',
    packages=find_packages(),
    version='0.3.7',
    description='A simple machine learning library based on Object Oriented design principles.',
    author='Shane Kercheval',
    author_email='shane.kercheval@gmail.com',
    license='MIT',
    url='https://github.com/shane-kercheval/oo-learning',
    download_url='https://github.com/shane-kercheval/oo-learning/archive/0.3.7.tar.gz',
    keywords=['machine-learning', 'data-science', 'object-oriented-programming', 'data-analysis'],
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3'],
    install_requires=[
        'bayesian-optimization>=1.2.0',
        'cython>=0.29.23',
        'dill>=0.3.3',
        'hdbscan>=0.8.27',
        'hyperopt>=0.2.5',
        'lightgbm>=3.1.1',
        'matplotlib>=3.3.4',
        'mock>=4.0.3',
        'numpy>=1.20.2',
        'pandas>=1.2.4',
        'scikit-learn>=0.24.2',
        'scipy>=1.6.2',
        'seaborn>=0.11.1',
        'statsmodels>=0.12.2',
        'xgboost>=1.4.2',
    ],
)
