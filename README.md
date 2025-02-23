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

Datasets used in previous cohort as examples:
Use Case: Real Estate Price Prediction
Dataset: House Prices (Kaggle)

We will build 3 ML services:

1️⃣ Real-time model: Price prediction using only user-inputted features.
2️⃣ Hybrid real-time model: Some features from the user, some from a database lookup.
3️⃣ Batch prediction service: Precomputed prices stored in an online table

https://www.kaggle.com/datasets/krantiswalke/bankfullcsv

https://www.kaggle.com/datasets/microize/newyork-yellow-taxi-trip-data-2020-2019

https://www.kaggle.com/code/nimapourmoradi/red-wine-quality

https://www.kaggle.com/datasets/gregorut/videogamesales

https://github.com/end-to-end-mlops-databricks-2/course-code-hub

# Data

Using the [**House Price Dataset**](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) from Kaggle.

This data can be used to build a classification model to calculate house price.

To create a new environment and create a lockfile, run:

```
uv venv -p 3.11 venv
source venv/bin/activate
uv pip install -r pyproject.toml --all-extras
uv lock
```

`uv pip install -e .` resolves the module not found issues
