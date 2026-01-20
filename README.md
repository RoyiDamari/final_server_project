# ML Model Training & Prediction Dashboard

This project is a full-stack machine learning platform that allows users to train models, make predictions, and analyze usage statistics through an interactive web interface.

---

## Overview

The system consists of:

* **FastAPI backend** for authentication, model training, predictions, and analytics
* **Streamlit frontend** for user interaction
* **PostgreSQL** for persistent data storage
* **Redis** for caching, rate-limiting, and performance optimization
* **Docker & Docker Compose** for reproducible deployment

---

## Features

### User & Security

* User registration and login
* JWT-based authentication with refresh tokens
* Token-based usage system (training, prediction, analytics)
* Rate-limiting per endpoint

### Model Training

* Upload CSV datasets
* Select features and label
* Train different model types:

  * Linear Regression
  * Logistic Regression
  * Random Forest
* Preset and custom hyperparameters
* Training metrics stored and displayed

### Predictions

* Select a trained model
* Dynamically generated input form based on feature schema
* Store and view prediction history

### Analytics Dashboard

* Distribution of model types
* Classification vs. regression split
* Label usage distribution
* Metric distributions (Accuracy, R²)

### Caching & Performance

* Redis-backed caching for expensive queries
* Versioned cache invalidation
* Token charging only when data changes

---

## Example Dataset

This repository includes a small example dataset:

```
data/exam_score_prediction.csv
```

This dataset can be used to:

* Train a regression model
* Test the full training → prediction workflow
* Verify that the application is working end-to-end

The dataset is optional and provided only as a convenience example.

---

## Using Other Datasets

You may use any compatible CSV dataset, including datasets from Kaggle:

[https://www.kaggle.com/datasets](https://www.kaggle.com/datasets)

**Dataset requirements:**

* CSV format
* One column used as the target label
* Remaining columns used as features
* Uploaded via the **Train Model** screen in the UI

The application does not rely on built-in datasets. All data is provided by the user at runtime.

---

## Prerequisites

To run this project you need:

* Docker
* Docker Compose
* Git

No local Python environment is required when using Docker.

---

## Environment Variables

Create a `.env` file based on the provided example:

```bash
cp .env.example .env
```

Example `.env` contents:

```env
# Database
POSTGRES_DB=app_db
POSTGRES_USER=app_user
POSTGRES_PASSWORD=strongpassword
DATABASE_URL=postgresql+asyncpg://app_user:strongpassword@postgres_db:5432/app_db

# Redis
REDIS_URL=redis://redis:6379/0

# API
API_BASE_URL=http://fastapi_app:8000

# Auth
SECRET_KEY=change_me
ALGORITHM=HS256

# OpenAI (optional)
OPENAI_API_KEY=
```

---

## Running the Application

From the project root:

```bash
docker compose up --build
```

This will start:

* PostgreSQL
* Redis
* FastAPI backend (port **8000**)
* Streamlit frontend (port **8501**)

---

## Accessing the Application

* **Streamlit UI:**
  [http://localhost:8501](http://localhost:8501)

* **FastAPI API:**
  [http://localhost:8000](http://localhost:8000)

---

## Typical Workflow

1. Open the Streamlit UI
2. Register a new user or log in
3. Upload a CSV dataset
4. Train a model
5. View training metrics
6. Make predictions
7. Explore usage dashboards

---

## Data Handling & Persistence

* Uploaded datasets are **not stored permanently**
* Trained models are stored on disk and referenced in the database
* Logs and models are mounted as Docker volumes
* Redis caches analytics and assist responses

---

## Notes on Data Files

* Only the example dataset is committed to Git
* Other datasets should be uploaded manually via the UI
* CSV/XLSX files are excluded from Docker images
* This keeps images clean and reproducible

---

## Stopping the Application

```bash
docker compose down
```

To also remove volumes (database, cache):

```bash
docker compose down -v
```

---

## License

This project is provided as-is for demonstration and learning purposes.
