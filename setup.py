from distutils.core import setup
from setuptools import find_packages
setup(
    name='oolearning',
    packages=find_packages(),
    version='0.2.83',
    description='A simple machine learning library based on Object Oriented design principles.',
    author='Shane Kercheval',
    author_email='shane.kercheval@gmail.com',
    license='MIT',
    url='https://github.com/shane-kercheval/oo-learning',
    download_url='https://github.com/shane-kercheval/oo-learning/archive/0.2.83.tar.gz',
    keywords=['machine-learning', 'data-science', 'object-oriented-programming', 'data-analysis'],
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3'],
    install_requires=[
        'hdbscan==0.8.15',
        'xgboost==0.71',
        'seaborn>=0.8.1',
        'scipy>=1.0.0',
        'mock>=2.0.0',
        'dill>=0.2.7.1',
        'scikit-learn>=0.19.1',
        'statsmodels>=0.8.0',
        'matplotlib>=2.1.2',
        'numpy>=1.14.0',
        'pandas>=0.22.0'
    ],
)
