from setuptools import setup, find_packages

setup(
    name="toy-cvrp-rl-solver",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
    ],
)
