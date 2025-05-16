# MLOps Project with GitHub Actions, GCP, Kubernetes, Docker, MLflow & DagsHub
## Overview
This project implements an end-to-end MLOps pipeline leveraging modern cloud-native tools.
It trains a machine learning model on Google Cloud Platform (GCP), tracks experiments with MLflow and DagsHub, and deploys the model via a Flask API running inside a Docker container on Kubernetes.

## 🚀 Features
- 📄 __Dataset__: Uses the Faulty Steel Plates dataset from Kaggle

- 🔍 __ML Pipeline__: Model training script runs on GCP using Docker containers

- 🤖 __Experiment Tracking__: MLflow integrated with DagsHub for versioning and experiment management

- ✂️ __CI/CD__: GitHub Actions workflow automates build, test, push, and deploy processes

- 💬 __Deployment__: Containerized Flask API deployed on GKE (Google Kubernetes Engine)

- ⚙️ __Secrets & Config__: Managed via GitHub Secrets and Kubernetes Secrets

## 🛠️ How It Works
__Training__: When the Docker container starts, it runs pipeline/training_pipeline.py which trains the model on the dataset, logging results to MLflow.

__Serving__: After training, the Flask API (app/application.py) starts and serves predictions via HTTP.

__Deployment__: Kubernetes manages the running containers and exposes the API externally.

__Tracking__: MLflow and DagsHub provide experiment tracking, model registry, and collaboration.

Note: You can look training metrics from dagshub https://dagshub.com/ckryusuff2/mlops-github-actions-gcp/experiments
