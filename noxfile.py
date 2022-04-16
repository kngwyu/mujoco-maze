import pathlib

import nox

SOURCES = ["src/mujoco_maze", "tests"]


def _install_self(session: nox.Session) -> None:
    session.install("setuptools", "--upgrade")
    session.install("-e", ".")


@nox.session(reuse_venv=True, name="pip-compile")
def pip_compile(session: nox.Session) -> None:
    session.install("pip-tools")
    requirements_dir = pathlib.Path("requirements")
    for path in requirements_dir.glob("*.in"):
        txt_file = f"requirements/{path.stem}.txt"
        session.run("pip-compile", path.as_posix(), "--output-file", txt_file)


@nox.session(reuse_venv=True, python=["3.7", "3.8", "3.9", "3.10"])
def tests(session: nox.Session) -> None:
    _install_self(session)
    session.install("-r", "requirements/test.txt")
    session.run("pytest", "tests", *session.posargs)


@nox.session(reuse_venv=True, python=["3.8", "3.9", "3.10"])
def lint(session: nox.Session) -> None:
    session.install("-r", "requirements/lint.txt")
    session.run("flake8", *SOURCES)
    session.run("black", *SOURCES, "--check")
    session.run("isort", *SOURCES, "--check")


@nox.session(reuse_venv=True)
def format(session: nox.Session) -> None:
    session.install("-r", "requirements/format.txt")
    session.run("black", *SOURCES)
    session.run("isort", *SOURCES)
