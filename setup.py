import io
import os
import re

from setuptools import find_packages, setup

NAME = "mujoco-maze"
AUTHOR = "Yuji Kanagawa"
EMAIL = "yuji.kngw.80s.revive@gmail.com"
URL = "https://github.com/kngwyu/mujoco-maze"
REQUIRES_PYTHON = ">=3.6.0"
DESCRIPTION = "Simple maze environments using mujoco-py"

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, "mujoco_maze/__init__.py"), "rt", encoding="utf8") as f:
    VERSION = re.search(r"__version__ = \"(.*?)\"", f.read()).group(1)

try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        LONG_DESCRIPTION = "\n" + f.read()
except FileNotFoundError:
    LONG_DESCRIPTION = DESCRIPTION


REQUIRED = ["gym>=0.16.0", "mujoco-py>=1.5.0"]


setup(
    name=NAME,
    version=VERSION,
    url=URL,
    project_urls={
        "Code": URL,
        "Issue tracker": URL + "/issues",
    },
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=REQUIRES_PYTHON,
    install_requires=REQUIRED,
    license="Apache2",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
)
