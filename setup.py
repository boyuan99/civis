from setuptools import setup, find_packages
import os
import re


def get_version():
    init_py = open(os.path.join('civis', '__init__.py')).read()
    return re.search("__version__ = ['\"]([^'\"]+)['\"]", init_py).group(1)


with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="civis",
    version=get_version(),
    packages=find_packages(exclude=['templates', '.idea']),
    include_package_data=True,
    install_requires=requirements,
    author="Bo Yuan",
    author_email="boy6@illinois.edu",
    description="A visualization server for Calcium Imaging data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/boyuan99/civis",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    entry_points={
        'console_scripts': [
            'civis=civis.app:main',
        ],
    },
)
