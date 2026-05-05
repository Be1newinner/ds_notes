# MLOps Submodule Map

This document outlines the structured teaching path for the MLOps module. 

## 1. Model Deployment (`01_model_deployment`)
- **Why it is taught:** To teach students how to take a model out of a Jupyter Notebook and make it accessible to other applications or users via web interfaces and APIs.
- **Content Type:** Code-heavy. Requires practical demonstrations of creating APIs.
- **Focus:** Building REST APIs with FastAPI and interactive web apps with Streamlit.

## 2. Model Tracking (`02_model_tracking`)
- **Why it is taught:** To solve the "which parameters gave me the best accuracy yesterday?" problem. Students need to learn structured experiment management.
- **Content Type:** Code-heavy and visual. Requires showing the MLflow UI.
- **Focus:** Using MLflow to log parameters, metrics, and models.

## 3. Containerization (`03_containerization`)
- **Why it is taught:** "It works on my machine" is a common problem in Data Science. Containerization solves the dependency and environment issues.
- **Content Type:** Code-heavy. Requires writing Dockerfiles and using Docker commands.
- **Focus:** Building and running Docker images for ML applications.

## 4. Model Monitoring (`04_model_monitoring`)
- **Why it is taught:** Models degrade over time. Students must understand that deployment is not the end of the lifecycle.
- **Content Type:** Theory-heavy with some visual/code examples.
- **Focus:** Understanding Data Drift, Concept Drift, and strategies for retraining models.

## Recommended Order of Teaching
1. **Model Deployment:** Start by making the model useful immediately. It's exciting for students to see their model running as a web app.
2. **Containerization:** Now that we have an app, how do we share it reliably? Introduce Docker.
3. **Model Tracking:** When we deploy models, we need to know which version it is. Introduce MLflow for managing the ML lifecycle.
4. **Model Monitoring:** Finally, discuss what happens when the containerized, tracked model is out in the wild.
