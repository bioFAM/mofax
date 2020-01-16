import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mofax-gtca",
    version="0.0.1",
    author="Danila Bredikhin",
    author_email="danila.bredikhin@embl.de",
    description="Load and interpret MODE models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gtca/mofax",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'h5py'
    ]
)
