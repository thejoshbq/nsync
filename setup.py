from setuptools import setup, find_packages

setup(
    name="nsync2p",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "seaborn",
        "tqdm",
        "scikit-learn",
    ],
    author="Josh Boquiren",
    author_email="thejoshbq@proton.me",
    description="A package for neural data synchronization with behavioral events and population analysis",
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
)