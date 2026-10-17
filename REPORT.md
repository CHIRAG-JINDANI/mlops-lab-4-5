# Lab Assignment 4–5: Advanced MLflow & Data Versioning with DVC

* **Student:** Chirag Jindani
* **Roll No:** 24AI020
* **Course:** Machine Learning Operations (MLOps)
* **Repository:** [https://github.com/CHIRAG-JINDANI/mlops-lab-4-5](https://github.com/CHIRAG-JINDANI/mlops-lab-4-5)

---

## 1. Project Overview & Objective

In traditional software development, Git tracks everything needed to build and run an application. In machine learning, tracking the code alone isn't enough—the model's behavior depends heavily on the dataset version, random seeds, hyperparameters, and the exact training environment.

This lab focuses on answering one central question:
> **"Can another engineer identify which model is currently running in production, and easily retrieve the exact code and dataset version used to build it?"**

To solve this, I built an end-to-end reproducible setup using three tools working together:
1. **Git:** Versions the source code, configs, and lightweight metadata files (`.dvc`, `dvc.yaml`, `dvc.lock`).
2. **DVC:** Handles the dataset versioning outside Git, caching data files in a local remote storage folder and letting us roll back data with simple checkout commands.
3. **MLflow:** Tracks training runs, logs parameters and metrics ($MAE, RMSE, R^2$), saves visual artifacts, enforces input/output model signatures, and manages the model lifecycle in the Model Registry.

For the dataset, I used the **California Housing dataset** (predicting median house values in \$100k units), training three different candidate architectures and tracking their lineage from start to finish.

---

## 2. Part A: Advanced MLflow (Lab 4)

### A1. Key MLflow Concepts Explained

* **Model Signature:**
  * *What it is:* A clear schema definition of what inputs a model expects (column names and data types like `double` or `string`) and what output it produces (e.g. continuous float or tensor).
  * *Practical Example:* For our California housing models, the signature specifies that the model requires 8 numeric columns (`MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`) and outputs a 1D array of `float64`. If someone sends a request missing the `Latitude` field or passes a string instead of a float, MLflow catches this immediately with an HTTP 400 error before running `.predict()`.
* **Artifacts:**
  * *What they are:* Files generated during an experiment run that you want saved alongside your metrics. This includes trained model weights (`model.pkl`), diagnostic plots, configuration files, and evaluation reports.
  * *In this lab:* For every run, I logged the serialized model pipeline, an evaluation summary JSON, and three plots: residuals plot, actual vs. predicted plot, and feature importances.
* **Autologging:**
  * *What it is:* A built-in MLflow feature (e.g. `mlflow.sklearn.autolog()`) that hooks into popular libraries and logs common parameters and metrics without writing lots of manual `mlflow.log_*` calls.
  * *What it recorded for us:* It automatically logged Scikit-learn parameters (like `alpha` for Ridge, `n_estimators`, `max_depth`), training times, and training metrics. However, autologging didn't capture our custom DVC data hashes, Git commit SHAs, or isolated test-set diagnostic plots, so I used a hybrid approach: autologging for estimator basics, plus explicit logging for our audit tags and evaluation artifacts.
* **Model Registry:**
  * *What it is:* A centralized model store where trained models can be registered under a friendly name (e.g. `HousePriceModel`) with automatic versioning (v1, v2, v3), descriptions, tags, and deployment statuses.
  * *Why it helps:* Instead of giving deployment engineers an obscure run ID like `d5f0fe90fb8a...`, we register the model and manage its lifecycle cleanly.
* **Version vs. Stage / Alias:**
  * A **Version** is permanent and immutable (e.g. Version 1, Version 2). Once created, its weights and metrics never change.
  * A **Stage** (classic MLflow: `Staging`, `Production`, `Archived`) or **Alias** (modern MLflow: `@candidate`, `@production`, `@archived`) is a movable label indicating where the model is currently used. While Version 3 stays Version 3 forever, the `@production` alias can be shifted from Version 2 to Version 3 once Version 3 passes validation.

---

### A2 & A3. Experiment Tracking & Candidate Models

I trained three candidate models with different trade-offs:
1. **Candidate 1 (`v1_ridge`):** Ridge regression baseline ($\alpha = 1.0$) with a `StandardScaler`. Simple, fast, and highly interpretable.
2. **Candidate 2 (`v2_random_forest`):** Random Forest Regressor (100 trees, `max_depth=10`). Captures non-linear relationships and geographic pockets.
3. **Candidate 3 (`v3_gradient_boosting`):** Gradient Boosting Regressor (200 trees, `learning_rate=0.08`, `max_depth=5`). Optimizes sequential errors for higher predictive accuracy.

Each run logged test evaluation metrics ($MAE$, $RMSE$, $R^2$) and saved two key visual diagnostics:
* **Residuals Plot (`residuals_plot.png`):** Checks whether errors are symmetrically distributed around zero or if there is systematic bias.
* **Actual vs. Predicted Plot (`actual_vs_predicted.png`):** Shows how closely predictions follow the ideal 45-degree line.

> 📸 **[SCREENSHOT 1: MLflow Runs Comparison Table]**  
> *How to take it: Run `.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`, open `http://127.0.0.1:5000`, click on Experiment `California_House_Price_Prediction`, and capture the runs table showing `candidate_v1_ridge`, `candidate_v2_random_forest`, and `candidate_v3_gradient_boosting` along with their test metrics and tags.*

---

### A4. Model Signatures & Input Examples

The model signature was inferred using `infer_signature(X_train, model.predict(X_train))` and logged alongside an input example (`X_train.head(5)`):
* **Inputs:** 8 `double` columns (`MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`).
* **Outputs:** 1D array of `float64` representing predicted house prices in \$100k units.

#### Discussion: Handling Bad Inputs
* **What happens if a future API sends 3 columns instead of 4, or changes a column type?**  
  Without a signature, Scikit-learn will typically crash with an ugly traceback like `ValueError: X has 3 features, but RandomForestRegressor is expecting 8 features`. Worse, if someone passes an unindexed NumPy array with columns in the wrong order, the model might make silent, completely wrong predictions without raising any error! With an MLflow signature attached, MLflow inspects the payload first. If a column is missing or types are mismatched, it immediately rejects the request with an HTTP 400 Bad Request and tells the client exactly which column failed.
* **What can a signature detect, and what can it NOT guarantee?**  
  * **What it catches:** Missing columns, extra columns, misspelled column names, and mismatched data types (e.g. passing `"five"` instead of `5.0`).
  * **What it cannot guarantee:** It cannot check if values make physical sense. For example, if someone passes a negative `HouseAge` ($-40$) or coordinates located in the middle of the Pacific Ocean, the signature sees valid floats and lets the request through. Detecting semantic drift or out-of-range values requires additional data validation tools (like Great Expectations or Pydantic validators).

> 📸 **[SCREENSHOT 2: Model Artifacts & Signature]**  
> *How to take it: In the MLflow UI, click on the `candidate_v3_gradient_boosting` run, scroll down to Artifacts, click the `model` folder, and take a screenshot of the Schema tab showing Inputs, Outputs, and the Input Example.*

---

### A5. Model Registry Versions (v1, v2, v3)

All three candidate models were registered under the registered model name **`HousePriceModel`**:

| Version | Source Run ID | Test RMSE | Test MAE | Test $R^2$ | Model Details | Decision |
| :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **Version 1** | `1d0d1063df0f...` | 0.7530 | 0.5315 | 0.5715 | Ridge Regression baseline with StandardScaler. | **Archived / Retired** (High error; baseline). |
| **Version 2** | `6f1b24b2c032...` | 0.5428 | 0.3639 | 0.7774 | Random Forest (100 trees, max depth 10). | **Staging / Candidate** (Meets criteria; good backup). |
| **Version 3** | `d5f0fe90fb8a...` | **0.4783** | **0.3175** | **0.8272** | Gradient Boosting (200 trees, lr 0.08, depth 5). | **Production** (Best overall accuracy; promoted). |

> 📸 **[SCREENSHOT 3: Model Registry Versions Overview]**  
> *How to take it: Click on the "Models" tab at the top of MLflow UI, click `HousePriceModel`, and take a screenshot showing Version 1 (Archived), Version 2 (Staging), and Version 3 (Production).*

---

### A6. Lifecycle Management: Staging, Production, and Archived

#### Promotion Acceptance Rule
In `configs/config.yaml`, I defined an explicit rule before promoting any model to production:
* **Rule:** A candidate must achieve **$\text{Test RMSE} \le 0.65$** (error under \$65,000) and **$R^2 \ge 0.70$**.

#### Lifecycle Status Record

| Lifecycle State | Version | Reason & Evidence | Action Taken |
| :--- | :---: | :--- | :--- |
| **Production** | **v3** | Test RMSE is **0.4783** (well below 0.65) and $R^2$ is **0.8272**. Best predictive performance on the test set. | Assigned alias `@production` and stage `Production`. |
| **Staging / Candidate** | **v2** | Test RMSE is **0.5428**, meeting the threshold. Retained as a validated fallback. | Assigned alias `@candidate` and stage `Staging`. |
| **Archived / Retired** | **v1** | Test RMSE is **0.7530**, failing the threshold due to linear underfitting. | Assigned alias `@archived` and stage `Archived`. |

#### How a Teammate Retrieves the Production Model
Anyone on the team can load the current production model without having to know which run or version ID is live:
```python
import mlflow.pyfunc

mlflow.set_tracking_uri("sqlite:///mlflow.db")
# Load model dynamically via alias
model = mlflow.pyfunc.load_model("models:/HousePriceModel@production")
predictions = model.predict(new_data)
```

#### Critical Thinking: Should the highest-accuracy model always be promoted?
**Not necessarily.** Picking a model solely on the lowest test RMSE can backfire for several practical reasons:
1. **Inference Latency & SLAs:** An ensemble of hundreds of deep trees or a huge neural net might take 60ms to predict, while a simpler model takes 2ms. If the user-facing API has a 20ms SLA, the "better" model can't be used.
2. **Subgroup Disparities:** A model might look great on average across California, but perform terribly on specific lower-income counties or rural zip codes. We need to evaluate subgroup metrics before shipping.
3. **Overfitting to Test Split:** Highly flexible models might be tuning into quirks of the test set that won't hold up in production.
4. **Different Costs of Errors:** In house pricing, underestimating a home value by \$100k might lose a customer, while overestimating might cause the property to sit on the market for months. If error costs are asymmetric, RMSE alone doesn't reflect real business impact.

---

## 3. Part B: Data Versioning with DVC (Lab 5)

### B1. Why DVC? Core Concepts

* **Why Git isn't suited for large datasets:** Git tracks every change by saving compressed snapshots inside `.git/objects`. If you track a 500 MB CSV and edit a few rows, Git saves another 500 MB blob. The repository quickly becomes huge, cloning takes forever, and services like GitHub reject files over 100 MB.
* **What DVC does instead:** DVC leaves Git to do what it's good at (tracking code and small text files) while DVC manages the heavy data. DVC computes a unique MD5 hash of your dataset, moves the raw file to cache or remote storage, and leaves behind a tiny 4-line text pointer file (e.g. `data/dataset.csv.dvc`).
* **Git commit vs. DVC version:** A Git commit snapshots your scripts and the `.dvc` text pointer. The DVC version is the actual physical file sitting in your remote storage matching that MD5 hash.
* **DVC remote:** A shared storage location (like a local folder, S3 bucket, or Google Drive) where team members push and pull datasets, so everyone has access to the exact data version without sending gigabytes through Git.

---

### B2 & B4. Setup & Local Remote Configuration

1. Initialized DVC inside the repository:
   ```bash
   dvc init
   ```
2. Set up a local directory remote outside the git repo at `../dvc-storage`:
   ```bash
   mkdir -p ../dvc-storage
   dvc remote add -d localremote ../dvc-storage
   git add .dvc/config
   git commit -m "Configure DVC remote"
   ```

---

### B5 & B6. Tracking, Modifying, and Recovering Data

To test data versioning and recovery:
1. **Dataset v1 (Baseline):** Generated the California Housing dataset with 20,640 rows.
   ```bash
   python src/prepare_data.py --version v1
   dvc add data/dataset.csv
   git add data/dataset.csv.dvc data/.gitignore
   git commit -m "Track dataset v1 with DVC"
   dvc push
   ```
2. **Dataset v2 (Controlled Evolution):** Simulated new survey entries by appending 100 realistic records (20,740 rows).
   ```bash
   python src/prepare_data.py --version v2
   dvc add data/dataset.csv
   git add data/dataset.csv.dvc
   git commit -m "Update dataset to v2 (added 100 survey samples)"
   dvc push
   ```

#### Dataset Version Comparison Table

| Version | Git Commit | Rows | Columns | File Size | DVC MD5 Hash | Change Description | Restored & Verified? |
| :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: |
| **Dataset v1** | `6767005` | 20,640 | 9 | 1,895,154 B | `ca5b757f0729c68d74d05bf9d9ca8486` | Baseline California Housing data. | **Yes** (Row count 20,640 & hash match). |
| **Dataset v2** | `666c6cd` | 20,740 | 9 | 1,903,729 B | `5968e75c88d31ad1a1083b00d57fba4e` | Appended 100 new survey samples. | **Yes** (Row count 20,740 verified on main). |

#### How I Verified the Restoration
To prove that DVC actually restored the old version:
1. Checked out the v1 commit into a temporary branch: `git checkout -b verify-v1 6767005`
2. Ran `dvc checkout` to pull the matching data from cache.
3. Ran `wc -l data/dataset.csv` $\rightarrow$ exactly **20,641 lines** (1 header + 20,640 data rows).
4. Switched back to `main` (`git checkout main`) and ran `dvc checkout` $\rightarrow$ verified row count returned to **20,741 lines**.

*Why filename alone isn't enough:* A file named `dataset.csv` could have rows added, deleted, or corrupted without the filename changing at all. Only cryptographic checksums (MD5) linked with Git commits guarantee you're training on the exact same data.

---

### B7. DVC Pipelines (`dvc.yaml`)

I created a pipeline stage in `dvc.yaml` that connects the dataset to model training:
```yaml
stages:
  train_and_evaluate:
    cmd: python3 src/train.py --candidate all
    deps:
      - configs/config.yaml
      - data/dataset.csv
      - src/train.py
      - src/evaluate.py
    outs:
      - reports/figures:
          cache: false
```

When you run `dvc dag`, DVC displays the dependency flow:
```
+----------------------+ 
| data/dataset.csv.dvc | 
+----------------------+ 
            *            
            *            
            *            
 +--------------------+  
 | train_and_evaluate |  
 +--------------------+  
```

When running `dvc repro`, DVC executes the training stage and saves the exact dependency hashes to `dvc.lock`. If you run `dvc repro` again without modifying the data, config, or code, DVC skips execution:
```
Stage 'train_and_evaluate' didn't change, skipping
Data and pipelines are up to date.
```

---

## 4. Part C: Integrating Git, DVC, and MLflow

### C1. Tool Responsibilities Summary

| Tool | What It Tracks | Where It Lives | Why It Matters |
| :--- | :--- | :--- | :--- |
| **Git** | Code, configs, and `.dvc` pointer files. | `.git/` folder and GitHub. | Gives a clear commit history connecting code to data pointers. |
| **DVC** | The raw dataset files and pipeline DAG. | `.dvc/cache` and `../dvc-storage`. | Stops large datasets from bloating Git while ensuring exact data rollbacks. |
| **MLflow** | Experiment metrics, parameters, artifacts, and model stages. | `sqlite:///mlflow.db` and `mlartifacts/`. | Tracks experiment history, logs signatures, and manages the `@production` model. |

---

### C2. End-to-End Lineage: Audit Tags in MLflow

To make sure everything is connected, `src/train.py` automatically injects the current Git commit SHA and DVC dataset MD5 hash into every MLflow run as tags:
```python
mlflow.set_tag("git_commit_sha", get_git_commit_sha())
mlflow.set_tag("dvc_dataset_hash", get_dvc_data_hash())
mlflow.set_tag("model_variant", candidate_key)
mlflow.set_tag("dataset_rows", str(len(df)))
```

> 📸 **[SCREENSHOT 4: MLflow Tags with Git and DVC Hashes]**  
> *How to take it: Click on the `candidate_v3_gradient_boosting` run in the MLflow UI, expand the "Tags" section, and capture the tags showing `git_commit_sha`, `dvc_dataset_hash`, and `dataset_rows`.*

---

### C3. Scenario Walkthrough: Debugging a Production Anomaly

Imagine a client flags an odd prediction from our production model on a house in San Francisco. Here is the exact step-by-step process any engineer can follow to investigate:

```
[Production Model: HousePriceModel@production]
                     │
                     ▼
          MLflow Version 3 (Run ID: d5f0fe90...)
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
Git SHA: 666c6cd8        DVC Hash: 5968e75c
         │                       │
         ▼                       ▼
 git checkout 666c6cd8      dvc checkout
         │                       │
         └───────────┬───────────┘
                     ▼
        [Identical Code & Data Restored]
                     ▼
      python src/evaluate.py --model-uri ...
```

1. **Find the live model:** Query the registry: `models:/HousePriceModel@production` $\rightarrow$ identifies Version 3 (Run ID `d5f0fe90fb8a...`).
2. **Read the audit tags:** Look at the run's tags in MLflow:
   * `git_commit_sha`: `666c6cd8204c03ae33546720b162db4fbdc6bca2`
   * `dvc_dataset_hash`: `5968e75c88d31ad1a1083b00d57fba4e`
3. **Restore the code:** Run `git checkout 666c6cd8204c...`
4. **Restore the dataset:** Run `dvc checkout` (DVC reads the pointer and retrieves the exact 20,740-row CSV from cache).
5. **Reproduce the prediction:** Run `python src/evaluate.py --model-uri "models:/HousePriceModel@production"`.  
   We are now running the exact model weights on the exact data it was trained on, with zero guesswork.

---

## 5. Real Challenges Encountered & How I Fixed Them

1. **Python 3.15 Wheel Incompatibilities:**  
   * *Problem:* When first setting up the virtual environment with the default Python 3.15 binary, `pip install pyarrow pygit2` failed because Python 3.15 is very new and pre-compiled wheels were not available on PyPI, causing it to fail while trying to compile `libgit2` from source.
   * *Fix:* Created the virtual environment using Homebrew's Python 3.14 (`/opt/homebrew/bin/python3.14`), where pre-compiled wheels for macOS Apple Silicon installed cleanly.
2. **MLflow 3.x Tree Serialization Security Warning:**  
   * *Problem:* Logging Scikit-learn tree models with `mlflow.sklearn.log_model()` failed with `skops.io.exceptions.UntrustedTypesFoundException: Untrusted types found: ['sklearn.tree._tree.Tree']`. MLflow 3.x defaults to `skops` format which restricts unpickling tree nodes.
   * *Fix:* Added `serialization_format="cloudpickle"` inside `log_model()`, allowing standard serialized tree ensembles to save without errors.
3. **DVC Pipeline Output Clash:**  
   * *Problem:* When setting up `dvc.yaml`, DVC threw `ERROR: output 'data/dataset.csv' is specified in data/dataset.csv.dvc and prepare stage`. DVC doesn't allow a file to be both tracked by a `.dvc` file and produced by a `dvc.yaml` stage.
   * *Fix:* Kept `data/dataset.csv.dvc` as the data tracker and made `data/dataset.csv` a dependency (`deps`) of the training stage, writing stage outputs to `reports/figures`.
4. **Model Registry SQLite Requirement:**  
   * *Problem:* MLflow's default file-based storage (`./mlruns`) doesn't support the full Model Registry with aliases and stage transitions.
   * *Fix:* Configured `sqlite:///mlflow.db` as the tracking URI, which provides full Model Registry features locally on disk.

---

## 6. Conclusion

Setting up Git, DVC, and MLflow together took some initial plumbing, but the payoff is huge. Instead of having messy folders with names like `model_final_v2_really_final.pkl` and `data_clean_new.csv`, we have:
* An immutable history of every dataset change in DVC.
* An experiment tracking dashboard in MLflow with metrics, plots, and signatures.
* An unambiguous Model Registry where `@production` always points to the best validated model.
* Total traceability: any production model can be traced back to its Git commit and dataset hash in minutes.

---

## 7. References

1. MLflow Documentation: [https://mlflow.org/docs/latest/](https://mlflow.org/docs/latest/)
2. DVC Documentation: [https://dvc.org/doc](https://dvc.org/doc)
3. Scikit-learn California Housing: [https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
4. D. Sculley et al., *"Hidden Technical Debt in Machine Learning Systems"*, NeurIPS 2015.
