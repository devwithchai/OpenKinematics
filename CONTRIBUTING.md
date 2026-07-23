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

[packaging] add pyproject.toml and editable initial README

[packaging] untrack generated egg-info, expand gitignore
```

Prefer several small, focused commits over one large commit that mixes
unrelated changes.

---

## Coding Style

OpenKinematics follows standard Python coding conventions with an emphasis on
clarity and maintainability.

When contributing code:

- Follow PEP 8 style guidelines.
- Prefer readability over clever or highly condensed implementations.
- Use descriptive variable, function, and class names.
- Keep functions focused on a single responsibility.
- Prefer explicit code over hidden abstractions or unnecessary complexity.
- Keep implementations consistent with the surrounding codebase.

Code should be easy to understand for contributors who are learning robotics
software as well as experienced developers.

---

## Project Architecture

## Project Architecture

OpenKinematics follows a layered architecture.

```text
api
 ↓
robots
 ↓
solvers
 ↓
math
```

Each layer has a clearly defined responsibility.

- `math/` contains mathematical primitives and matrix operations.
- `solvers/` implements forward kinematics, inverse kinematics, and Jacobian algorithms.
- `robots/` models robot-specific behaviour using the solver layer.
- `api.py` provides the public user-facing interface.

The remaining components support this architecture:

- `visualization/` depends on the `robots/` layer to obtain robot state for
  plotting, but is not part of the primary computational dependency chain. It
  is used by the public API to provide visualization utilities.
- `exceptions.py` defines shared exception types used throughout the project
  and is intentionally dependency-light so it can be imported by all layers.

When contributing:

- Do not introduce upward imports.
- Avoid circular dependencies.
- Keep layers independent.
- Preserve the existing architecture whenever possible.

---

## Denavit–Hartenberg Convention

OpenKinematics follows **Craig's Denavit–Hartenberg (DH) convention** throughout
the project.

DH parameters are always ordered as:

```text
(theta, d, r, alpha)
```

Contributors should follow these conventions consistently:

- Angles are expressed in **radians**.
- Homogeneous transformations are represented by **4×4 NumPy arrays**.
- Robot implementations should remain consistent with the existing DH
  parameter ordering and transformation conventions.

---

## Documentation Standards

Public modules, classes, and functions should include accurate
Sphinx-style docstrings.

Documentation should remain synchronized with the implementation.

When writing docstrings:

- Use Sphinx directives such as `:param:`, `:return:`, and `:raises:`.
- Use one `:raises:` entry per exception type, combining multiple causes into
  a single description where appropriate.
- Document propagated exceptions when they form part of the public API
  contract.
- Avoid repeating information already expressed by Python type hints.
- Keep descriptions concise, accurate, and implementation-specific.

Private helper functions generally do not require docstrings unless they
contain non-obvious logic that benefits future maintainers.

---

## Testing Expectations

Run the complete test suite before submitting changes.

```bash
pytest
```

Contributors should ensure that:

- Existing tests continue to pass.
- New functionality includes appropriate unit tests.
- Bug fixes include regression tests whenever practical.
- Tests remain readable and focused on observable behaviour.

Changes that modify public APIs or numerical algorithms should be accompanied
by appropriate test coverage.

---

## Public API Stability

OpenKinematics aims to provide a stable public API.

When contributing:

- Preserve backward compatibility whenever practical.
- Avoid breaking public interfaces without prior discussion.
- Prefer extending existing APIs over changing their behaviour.
- Update documentation whenever public behaviour changes.

---

## Pull Request Guidelines

Although OpenKinematics is currently maintained by a single developer,
future community contributions are expected to follow a pull request workflow.

When opening a pull request:

- Keep the change focused on a single logical topic.
- Clearly explain the motivation for the change.
- Include tests when adding new functionality or fixing bugs.
- Update documentation when public behaviour changes.
- Respond constructively to review feedback.

Repository automation (such as continuous integration) may expand over time,
so contributors should ensure their changes are ready for automated testing.

---

## Contributor Checklist

Before submitting a contribution, verify that:

- [ ] The project builds successfully.
- [ ] All tests pass.
- [ ] Public APIs are documented where appropriate.
- [ ] Documentation has been updated if behaviour changed.
- [ ] Examples have been updated if necessary.
- [ ] The change is contained in one logical commit or a small set of focused commits.