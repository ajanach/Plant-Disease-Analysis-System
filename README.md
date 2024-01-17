# Plant Disease Analysis System

## Overview:

This project focuses on developing a system for image analysis in agriculture, specifically targeting the detection of potato plant diseases based on leaf images using Convolutional Neural Networks (CNN).

## Features:

- **Model Training:** Utilizes CNN algorithms for training the disease detection model.
- **Model Serving:** Deploys the trained model using TensorFlow Serving for efficient and scalable serving.
- **FastAPI Backend:** Implements a FastAPI backend server that interacts with the TensorFlow Serving service.
- **Web UI Interface:** Provides a web interface for users to upload potato leaf images, facilitating communication with the backend for disease analysis.
- **Dockerized Deployment:** Creates Docker images for seamless deployment, ensuring consistency across environments.

## Azure Kubernetes Service (AKS) Architecture:

### Infrastructure Deployment using Terraform:

- **AKS Deployment:** Utilizes Terraform for deploying the Azure Kubernetes Service to facilitate a scalable and managed Kubernetes environment.

### Application Services Deployment:

- **TensorFlow Serving:** Sets up an application service within AKS to host the TensorFlow Serving instance.
- **FastAPI Backend:** Deploys a FastAPI backend service, connecting to the TensorFlow Serving for disease prediction.
- **Web UI with Nginx:** Configures an Nginx-powered application service to host the Web UI, making it publicly accessible.

### Ingress Configuration:

- **Ingress-Nginx Setup:** Establishes and configures Ingress-Nginx to manage external access to the Web UI, ensuring proper routing.

## Deployment Process:

1. **Training and Serving:** Train the model using CNN, serve it using TensorFlow Serving.
2. **Backend Deployment:** Deploy the FastAPI backend service to handle requests from the Web UI.
3. **Web UI Deployment:** Host the Web UI in Nginx, providing a user-friendly interface for uploading images.
4. **Ingress Configuration:** Configure Ingress-Nginx to make the Web UI publicly accessible.

## Future Work:

- Continuous improvement of the model through ongoing training.
- Enhancements to the user interface and additional features for a more comprehensive agricultural analysis system.
