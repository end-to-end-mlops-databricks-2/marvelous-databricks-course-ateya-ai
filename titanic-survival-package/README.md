# Titanic Survival Prediction Model

[![check.yml](https://github.com/ateya-ai/titanic-survival/actions/workflows/check.yml/badge.svg)](https://github.com/ateya-ai/titanic-survival/actions/workflows/check.yml)
[![publish.yml](https://github.com/ateya-ai/titanic-survival/actions/workflows/publish.yml/badge.svg)](https://github.com/ateya-ai/titanic-survival/actions/workflows/publish.yml)
[![Documentation](https://img.shields.io/badge/documentation-available-brightgreen.svg)](https://ateya-ai.github.io/titanic-survival/)
[![License](https://img.shields.io/github/license/ateya-ai/titanic-survival)](https://github.com/ateya-ai/titanic-survival/blob/main/LICENCE.txt)
[![Release](https://img.shields.io/github/v/release/ateya-ai/titanic-survival)](https://github.com/ateya-ai/titanic-survival/releases)

TODO.

# Installation

Use the package manager [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

# Usage

```bash
uv run titanic-survival
```

<h1 align="center">
Marvelous MLOps End-to-end MLOps with Databricks course

## Practical information

- Weekly lectures on Wednesdays 16:00-18:00 CET.
- Code for the lecture is shared before the lecture.
- Presentation and lecture materials are shared right after the lecture.
- Video of the lecture is uploaded within 24 hours after the lecture.

- Every week we set up a deliverable, and you implement it with your own dataset.
- To submit the deliverable, create a feature branch in that repository, and a PR to main branch. The code can be merged after we review & approve & CI pipeline runs successfully.
- The deliverables can be submitted with a delay (for example, lecture 1 & 2 together), but we expect you to finish all assignments for the course before the 25th of November.

## Set up your environment

In this course, we use Databricks 15.4 LTS runtime, which uses Python 3.11.
In our examples, we use UV. Check out the documentation on how to install it: https://docs.astral.sh/uv/getting-started/installation/

To create a new environment and create a lockfile, run:

```
uv venv -p 3.11 venv
source venv/bin/activate
uv pip install -r pyproject.toml --all-extras
uv lock
```
