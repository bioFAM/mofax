import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mofax",
    version="0.3.2",
    author="Danila Bredikhin",
    author_email="danila.bredikhin@embl.de",
    description="Load and interpret MOFA models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gtca/mofax",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["numpy", "pandas", "matplotlib", "seaborn", "h5py"],
)
