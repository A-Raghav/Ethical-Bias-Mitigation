from setuptools import find_packages, setup

with open("./requirements.txt") as text_file:
    requirements = text_file.readlines()

requirements = list(map(lambda x: x.rstrip("\n"), requirements))
install_libraries = [x for x in requirements]

__version__ = "0.0.1"


setup(
    name="fairai",
    version=__version__,
    description="FairAI is a wrapper class built around AIF360 for detecting and mitigating ethical bias in ML models",
    author="Advanced Analytics",
    author_email="aseem.raghav@innovaccer.com",
    packages=find_packages(include=["fairai", "fairai.*"]),
    include_package_data=True,
    install_requires=install_libraries,
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
