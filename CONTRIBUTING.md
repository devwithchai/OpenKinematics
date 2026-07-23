# Contributing to OpenKinematics

Thank you for your interest in contributing to OpenKinematics!

OpenKinematics is an educational robotics library built from first principles.
The goal of this project is to provide clean, readable implementations of
robot kinematics while maintaining high standards for correctness,
maintainability, and documentation.

Whether you are fixing a bug, improving documentation, or implementing a new
feature, your contribution is appreciated.

---

## Project Philosophy

OpenKinematics prioritizes correctness, readability, and maintainability over
clever implementations.

Every contribution should improve the educational value of the project while
preserving a clean layered architecture, a stable public API, and consistent
documentation.

---

## Development Environment

### Requirements

- Python 3.10 or later
- Git
- A virtual environment (recommended)

Clone the repository:

```bash
git clone https://github.com/devwithchai/OpenKinematics.git

cd OpenKinematics
```

Create and activate a virtual environment:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

Install the project together with development dependencies:

```bash
pip install -e ".[dev]"
```

This installs OpenKinematics in editable mode together with the project's
development dependencies, including `pytest`.

---

## Repository Structure

```text
OpenKinematics/
│
├── open_kinematics/      # Library source code
│   ├── math/
│   ├── solvers/
│   ├── robots/
│   ├── visualization/
│   ├── api.py
│   └── exceptions.py
│
├── examples/             # Runnable example programs
├── tests/                # Unit tests
├── docs/                 # Project documentation
├── README.md
├── CONTRIBUTING.md
├── LICENSE
└── pyproject.toml
```

---

## Current Development Workflow

OpenKinematics is currently maintained by a single developer.

Development has historically taken place directly on the `main` branch, and the
project history reflects this workflow.

---

## Future Contribution Workflow

As the project grows and community contributions increase, external
contributors are encouraged to:

- Create a feature branch for each logical change.
- Keep changes focused on a single topic.
- Submit the change as a GitHub Pull Request.
- Respond to review feedback before merging.

This workflow is intended for future community contributions and is not yet
enforced by repository automation.

---

## Commit Message Convention

Keep each commit focused on a single logical change.

Use a scoped prefix to indicate the purpose of the commit.

Examples from this repository:

```text
[docs] rewrite README for project overview and quickstart

[docs] audit and improve robot docstrings

[docs] audit and improve API facade docstrings

[docs] audit and improve visualization docstrings

[license] add MIT license file

[packaging] create pyproject.toml

[packaging] untrack generated egg-info and expand gitignore
```

Avoid combining unrelated changes into a single commit whenever possible.