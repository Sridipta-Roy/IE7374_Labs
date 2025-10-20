# ☁️ GitHub Lab: End-to-End CI/CD with GCP & Cloud Run

This lab demonstrates a complete MLOps workflow using **GitHub Actions, Google Cloud Storage (GCS), Artifact Registry, and Cloud Run**. It covers automated training, testing, Docker build, and deployment with CI/CD best practices.

---

## 🧱 Project Structure

```
├── api/
│   └── app.py                  # FastAPI app for predictions
├── data_pipeline/
│   └── data_fetcher.py         # Fetches CSV from GCP Bucket (CI-safe)
├── model_pipeline/
│   ├── train.py                # Trains and saves model artifact
│   ├── evaluate.py             # Evaluates model with threshold gating
│   └── save.py                 # Saves model with versioning + logs metrics to GCS
├── tests/                      # Unit tests (mocked GCP)
├── workflows/
│   └── ci_cd.yml               # GitHub Actions workflow
└── requirements.txt            # Dependencies
```

---

## ⚙️ Prerequisites

1. **GCP Setup**

   ```bash
   gcloud auth login
   gcloud config set project <PROJECT_ID>
   gcloud services enable run.googleapis.com artifactregistry.googleapis.com storage.googleapis.com
   ```

2. **Create Artifact Registry Repository**

   ```bash
   gcloud artifacts repositories create mlops-lab-repo \
     --repository-format=docker \
     --location=us-east1
   ```

3. **Create a Cloud Storage Bucket**

   ```bash
   gsutil mb -l us-east1 gs://your-bucket-name
   ```

4. **Service Account Setup**

   ```bash
   gcloud iam service-accounts create github-mlops-sa

   gcloud projects add-iam-policy-binding <PROJECT_ID> \
     --member="serviceAccount:github-mlops-sa@<PROJECT_ID>.iam.gserviceaccount.com" \
     --role="roles/admin"

   gcloud iam service-accounts keys create key.json \
     --iam-account=github-mlops-sa@<PROJECT_ID>.iam.gserviceaccount.com
   ```

5. **Upload Data File to GCS**

   ```bash
   gsutil cp data/iris.csv gs://your-bucket-name/data/iris.csv
   ```
---

## 🔐 GitHub Secrets Configuration

In your repository, go to:
**Settings → Secrets and Variables → Actions** → **New Repository Secret**

Add the following:

| Secret Name      | Description                                       |
| ---------------- | ------------------------------------------------- |
| `GCP_SA_KEY`     | Content of your `key.json` (Service Account Key)  |
| `GCP_PROJECT_ID` | GCP Project ID (e.g., `github-labs-mlops`)        |
| `GCP_BUCKET`     | GCS bucket name                                   |
| `GCP_FILE_PATH`  | Path to CSV inside bucket (e.g., `iris/iris.csv`) |

---

## 🧪 Running Locally

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variables (.env):**

   ```env
   GCP_BUCKET=your-bucket
   GCP_FILE_PATH=iris/iris.csv
   ```

3. **Train model locally:**

   ```bash
   python -m model_pipeline.train
   ```

4. **Evaluate model:**

   ```bash
   python -m model_pipeline.evaluate
   ```

5. **Run API locally:**

   ```bash
   uvicorn api.app:app --reload
   ```

   Then open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to test the `/predict` endpoint.

---

## ⚡ GitHub Actions Workflow Execution

### 1️⃣ **Automatic Trigger**

The workflow runs automatically on every **push to `main`** branch.

### 2️⃣ **Manual Trigger**

You can manually start it from the **Actions** tab:

* Go to your repository → **Actions → CI/CD – Train → Build → Deploy**
* Click **Run workflow** → Select branch → ✅ Run

---

## 🧭 CI/CD Workflow Summary

| Step                  | Description                                              |
| --------------------- | -------------------------------------------------------- |
| 🧩 **Install & Lint** | Installs dependencies and runs `flake8`                  |
| 🧪 **Unit Tests**     | Runs mocked tests (no real GCP dependency)               |
| 🧠 **Train Model**    | Executes training and saves artifact                     |
| 🔑 **Auth**           | Authenticates to GCP using GitHub secret service account |
| 🧰 **Docker Build**   | Builds image and pushes to Artifact Registry             |
| ☁️ **Deploy**         | Deploys latest version to Cloud Run                      |

---

## ☁️ Cloud Run Deployment Output

After successful deployment, you’ll see:

```
Deployed service [mlops-lab-api] to [https://mlops-lab-api-<hash>-us-east1.run.app]
```

You can open the URL to test your `/predict` API live.

---


## 🧾 Summary

✅ **Automated CI/CD** with GitHub Actions
✅ **Model versioning & metrics logging** to GCS
✅ **Cloud Run deployment** with reproducible builds
✅ **Mocked tests** ensuring pipeline stability

