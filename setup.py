from codecs import open
import re

import setuptools

with open("risk/__init__.py", "r") as f:
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE
    ).group(1)

if not version:
    raise RuntimeError("Cannot find version information")


install_requirements = [
    "numpy",
    "pandas",
    "psweep",
]

setuptools.setup(
    name="maker-risk-model",
    version=version,
    author="Block Analitica",
    packages=setuptools.find_packages(include=["risk", "risk.*"]),
    python_requires=">=3.6",
    install_requires=install_requirements,
    license="MIT",
    zip_safe=False,
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Internet",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
