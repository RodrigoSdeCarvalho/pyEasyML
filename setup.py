from setuptools import setup, find_packages
import os

VERSION = '2.0.5'
DESCRIPTION = 'A python machine learning framework.'
LONG_DESCRIPTION = 'A framework made to aid in the development of end-to-end machine learning projects, with data preprocessing, ml models, feature selection, hyperparameter tuning and much more.'

# Setting up
setup(
    name="pyEasyML",
    version=VERSION,
    author="Rodrigo Santos de Carvalho",
    author_email="<rodrigosc2401@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['keras', 'numpy', 'pandas', 'scikit_learn', 'xgboost', 'click'],
    entry_points={
        'console_scripts': [
            'pyEasyML=pyEasyML.pyEasyMLcli:cli',
        ],
    },
    keywords=['python', 'AI', 'Machine Learning', 'Neural Networks'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
