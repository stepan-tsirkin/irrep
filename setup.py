import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="irrep",
    author="Stepan S. Tsirkin",
    author_email="stepan.tsirkin@ehu.eus",
    description="A tool to get symmetry proberties of ab-initio wavefunctions, irreduible representations and more.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
        "scipy>=1.0",
        "spglib>=1.14",
        "Click",
        "monty",
        "ruamel.yaml",
        "irreptables>=1.1.0",
        "fortio",
        "packaging"
    ],
    include_package_data=False,
    url="https://github.com/irreducible-representations/irrep.git",
    packages=["irrep"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    entry_points="""
        [console_scripts]
        irrep=irrep.cli:cli
    """,
)
