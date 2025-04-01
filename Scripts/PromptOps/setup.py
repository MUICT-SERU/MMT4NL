from setuptools import setup, find_packages
from setuptools.command.install import install
import os

class PostInstallCommand(install):
    """Post-installation to download SpaCy language models and Coreferee."""
    def run(self):
        install.run(self)
        os.system("python -m spacy download en_core_web_sm -q")
        os.system("python -m spacy download en_core_web_lg -q")
        os.system("python -m coreferee install en -q")

setup(
    name='PromptOps',
    version='0.1.0',
    description='A Python library for LLM testing',
    author='ChommakornS',
    author_email='chommakorn.son@gmail.com',
    packages=find_packages(),
    install_requires=[
        'openai==0.28',
        'pandas>=1.3.0,<2.0.0',
        'sentence-transformers',
        'scikit-learn',
        'langchain',
        'google-generativeai',
        'nltk',
        'PyDictionary',
        'textblob',
        'transformers==4.31.0',
        'torch',
        'llama-stack',
        'anthropic',
        'language-tool-python',
        'huggingface-hub',
        'coreferee',
        'spacy==3.5.0',
        'numpy==1.22',
        'tqdm',
        'ipywidgets',
        'openpyxl'
    ],
    cmdclass={
        'install': PostInstallCommand,
    },
)
