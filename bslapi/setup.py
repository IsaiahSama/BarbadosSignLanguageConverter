"""This will be the setup file to turn bslapi into an actual package"""

from setuptools import setup, find_packages

setup(
    name="bslapi",
    version="0.1",
    packages=find_packages(),
    author="Isaiah Carrington",
    description="A package containing the classes to expose APIs to interact with onnx model.",
    license="MIT",
    keywords="sign language",
)
