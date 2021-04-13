import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepcase",
    version="0.0.1",
    author="<ANONYMIZED>",
    author_email="<ANONYMIZED>",
    description="DeepCASE: Semi-Supervised Contextual Analysis of Security "
                "Events",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/<ANONYMIZED>/DeepCASE",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
