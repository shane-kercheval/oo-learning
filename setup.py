from distutils.core import setup
from setuptools import find_packages
setup(
    name='oolearning',
    packages=find_packages(),
    version='0.3.6',
    description='A simple machine learning library based on Object Oriented design principles.',
    author='Shane Kercheval',
    author_email='shane.kercheval@gmail.com',
    license='MIT',
    url='https://github.com/shane-kercheval/oo-learning',
    download_url='https://github.com/shane-kercheval/oo-learning/archive/0.3.6.tar.gz',
    keywords=['machine-learning', 'data-science', 'object-oriented-programming', 'data-analysis'],
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3'],
    install_requires=[
        'cython>=0.29.23',
        'hdbscan>=0.8.27',
        'xgboost>=1.4.2',
        'seaborn>=0.11.1',
        'scipy>=1.6.2',
        'mock>=4.0.3',
        'dill>=0.3.3',
        'scikit-learn>=0.24.2',
        'statsmodels>=0.12.2',
        'matplotlib>=3.3.4',
        'numpy>=1.20.2',
        'pandas>=1.2.4'
    ],
)
