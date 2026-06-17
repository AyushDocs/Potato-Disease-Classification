from setuptools import setup, find_packages

setup(
    name="potato-disease-classification",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.12.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "huggingface-hub>=0.16.0",
    ],
    author="Ayush Paudel",
    description="Potato disease classification using CNN",
)
