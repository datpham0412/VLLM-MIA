from setuptools import setup, find_packages

setup(
    name="llama_adapter_v21",
    version="0.1",
    packages=find_packages(),   # will pick up the `llama/` folder
    install_requires=[],        # we’ve already installed deps elsewhere
)
