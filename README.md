# ML Model Training & Prediction Dashboard

This project is a full-stack machine learning platform that allows users to train models, make predictions, and analyze usage statistics through an interactive web interface.

---

## Demo

- Full app walkthrough: [Watch Demo](https://youtu.be/toWZ9XEUf-M)

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
* Account deletion with cache and token cleanup

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

## Token Charging Logic

The platform uses a token-based billing model for selected actions.

### Main Rules
- Training a new model charges tokens only after the training completes successfully.
- Making a new prediction charges tokens only after the prediction completes successfully.
- Repeated identical requests can reuse prior results through idempotency logic and may return without charging again.
- Metadata and dashboard endpoints use version-based charging, so users are charged only when the underlying dataset version changes.
- Internal UI-support endpoints such as model-selection helpers are not token-charged.
- Token purchases increase user balance and are protected by idempotency keys to prevent duplicate purchases.

---

## System Architecture (Full MVC + DB)
```mermaid
flowchart TD
%% Users
U[User] --> UI[Streamlit Frontend]
%% Frontend requests
UI -->|Login/Register / Auth| AUTH_CTRL[Auth Controller]
UI -->|Train Model| TRAIN_CTRL[Train Model Controller]
UI -->|Predict| PREDICT_CTRL[Prediction Controller]
UI -->|Analytics / Usage| USAGE_CTRL[User Usage Controller]
UI -->|Assist / Explain Param| ASSIST_CTRL[Assist Controller]
UI -->|Token Buy / Balance| TOKEN_CTRL[Token Credit Controller]
UI -->|Delete Account| USER_CTRL[User Controller]
UI -->|Health Check| HEALTH_CTRL[Health Controller]
%% Controllers call Services
AUTH_CTRL --> AuthService
TRAIN_CTRL --> TrainModelService
PREDICT_CTRL --> PredictionService
USAGE_CTRL --> UserUsageService
ASSIST_CTRL --> AssistService
TOKEN_CTRL --> TokenCreditService
USER_CTRL --> UserService
%% Services to Database
AuthService --> PG[(PostgreSQL)]
TrainModelService --> PG
TrainModelService --> MODELS[(Model Files / Volumes)]
PredictionService --> PG
PredictionService --> MODELS
UserUsageService --> PG
UserUsageService --> REDIS[(Redis Cache)]
AssistService --> PG
TokenCreditService --> PG
TokenCreditService --> REDIS
UserService --> PG
%% Health checks
HEALTH_CTRL --> PG
HEALTH_CTRL --> REDIS
%% Redis for caching & rate-limiting
TRAIN_CTRL --> REDIS
PREDICT_CTRL --> REDIS
%% Docker Environment
subgraph Docker
    UI
    AUTH_CTRL
    TRAIN_CTRL
    PREDICT_CTRL
    USAGE_CTRL
    ASSIST_CTRL
    TOKEN_CTRL
    USER_CTRL
    HEALTH_CTRL
    PG
    REDIS
    MODELS
end
%% DB Relationships (simplified)
subgraph DB_Tables
    USER_T[Users]
    TRAINED[TrainedModels]
    PREDICT[Predictions]
    TOKENS[TokenCredits]
    SESS[AuthSessions]
    SEEN[SeenVersions]
end
PG --> USER_T
PG --> TRAINED
PG --> PREDICT
PG --> TOKENS
PG --> SESS
PG --> SEEN
USER_T --> TRAINED
USER_T --> PREDICT
USER_T --> TOKENS
USER_T --> SESS
USER_T --> SEEN
TRAINED --> PREDICT
```

### Notes:
- This diagram shows **all controllers → services → DB/Redis/files**.
- **User → models → predictions → token credits** relationships are explicitly shown.
- Async training is implied in the TrainModelController → TrainModelService → MODELS path.
- Redis is shown for caching and rate-limiting.
- Docker box shows everything containerized.

---

## Train Model Sequence Diagram
```mermaid
sequenceDiagram
    participant U as User
    participant UI as Streamlit UI
    participant CTRL as Train Controller
    participant SVC as Train Service
    participant WORKER as Training Worker
    participant DB as PostgreSQL
    participant FS as Model Files
    participant R as Redis

    U->>UI: Upload CSV and choose model settings
    UI->>CTRL: Send train request
    CTRL->>SVC: Start training flow
    SVC->>SVC: Validate inputs and prepare fingerprint
    SVC->>DB: Create or reuse training row
    SVC->>WORKER: Run training subprocess
    WORKER->>FS: Write trained model artifact
    WORKER-->>SVC: Return metrics
    SVC->>DB: Mark row applied and charge token
    SVC->>R: Invalidate models cache version
    SVC-->>CTRL: Return trained model result
    CTRL-->>UI: Show metrics and updated balance
```

### Notes:
- Before training starts, the backend validates the CSV, selected features, label, model type, and hyperparameters.
- A deterministic training fingerprint is computed so identical requests can be detected and handled idempotently.
- The database row lifecycle uses PENDING, APPLIED, and FAILED states.
- Heavy model training runs in a subprocess so the main application does not keep a long database transaction open.
- The trained artifact is written to a temporary path first and then moved atomically to its final path.
- Tokens are charged only after successful training is completed and the row is marked APPLIED.
- If the same training request already succeeded earlier, the existing result can be returned without charging again.
- After success, the global models metadata cache is invalidated through Redis versioning.

---

## Prediction Sequence Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant UI as Streamlit UI
    participant CTRL as Prediction Controller
    participant SVC as Prediction Service
    participant DB as PostgreSQL
    participant FS as Model Files
    participant R as Redis

    U->>UI: Select model and enter feature values
    UI->>CTRL: Send prediction request
    CTRL->>SVC: Start prediction flow
    SVC->>DB: Load applied model metadata
    SVC->>FS: Load trained model artifact
    SVC->>SVC: Validate features and prepare fingerprint
    SVC->>DB: Create or reuse prediction row
    SVC->>SVC: Run prediction
    SVC->>DB: Mark row applied and charge token
    SVC->>R: Invalidate predictions cache version
    SVC-->>CTRL: Return prediction result
    CTRL-->>UI: Show prediction and updated balance
```

### Notes:
- The backend verifies that the selected trained model belongs to the user and is already in APPLIED status.
- The trained artifact is loaded from disk before prediction execution.
- A deterministic prediction fingerprint is computed from the model and input values for idempotency.
- The feature keys provided by the user must match the trained model’s expected feature schema.
- Prediction execution runs with timeout protection to avoid hanging requests.
- Tokens are charged only after successful prediction and state transition to APPLIED.
- If the same prediction request already succeeded, the stored result can be returned without charging again.
- After success, the global predictions metadata cache is invalidated through Redis versioning.

---

## Engineering Highlights

- Full-stack ML platform built with Streamlit, FastAPI, PostgreSQL, Redis, and Docker
- Strategy-pattern machine learning architecture for extensible model training flows
- Async backend workflows with subprocess-based training for CPU-heavy tasks
- Idempotent training, prediction, and token-purchase flows using deterministic fingerprints and unique constraints
- Persistent state lifecycle management with `PENDING`, `APPLIED`, and `FAILED` rows
- Atomic temp-to-final artifact publishing for trained model files
- Versioned Redis cache invalidation for global metadata endpoints
- Startup reconciliation logic for crash recovery and filesystem/database consistency
- JWT authentication with refresh-token rotation and session tracking
- Centralized exception handling and structured application logging

---

## Technical Design Highlights

- MVC-style backend separation using controllers, services, repositories, ORM models, and Pydantic schemas
- Shared ML base strategy with reusable preprocessing, cross-validation selection, and task validation
- Automatic regression/classification compatibility checks based on target type
- Selective skew handling for numeric features using log1p and Yeo-Johnson transforms
- Dynamic preprocessing for numeric and categorical columns through sklearn pipelines
- Deterministic fingerprinting for deduplication, idempotency, and safe retries
- Short database transactions around state transitions, with heavy compute moved outside transactions
- Crash-safe startup reconciler for pending rows, partial publishes, temp files, and orphan artifacts
- Redis-backed rate limiting and metadata caching with version-based invalidation
- Clear separation between user-facing errors and internal logged error details

---

## Key Backend Features

- User registration, login, logout, refresh-token rotation, and account deletion
- Token-based usage and idempotent token purchase flow
- Upload-based model training with configurable features, label, and hyperparameters
- Prediction flow using persisted trained model artifacts
- User-specific and global model/prediction history endpoints
- Analytics endpoints for model type distribution, problem type split, label distribution, and metric distribution
- Health/readiness checks for disk, PostgreSQL, and Redis
- Assist endpoint for parameter explanations and free-text ML questions
- Structured logging, centralized exception handling, and Redis-based rate limiting
- Dockerized deployment with persistent volumes for logs and trained artifacts

---

## UI Screenshots

### Authentication
- Register/Login: [View screenshot](screenshots/register_login.png)

### Billing
- Buy Tokens: [View screenshot](screenshots/buy_tokens.png)

### Training
- Model Results: [View screenshot](screenshots/model_results.png)
- Model Viewer: [View screenshot](screenshots/model_viewer.png)

### Prediction
- Prediction Results: [View screenshot](screenshots/prediction_results.png)
- Prediction Viewer: [View screenshot](screenshots/prediction_viewer.png)

### User Tokens Dashboard
- Tokens Dashboard: [View screenshot](screenshots/user_tokens_dashboard.png)

### User Usage Dashboard
- Model Type Distribution: [View screenshot](screenshots/model_type_distribution.png)
- Problem Type Distribution: [View screenshot](screenshots/problem_type_distribution.png)
- Label Distribution: [View screenshot](screenshots/label_distribution.png)
- Performance Trends: [View screenshot](screenshots/performance_trends.png)

### Account Management
- Delete Account: [View screenshot](screenshots/delete_account.png)

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
