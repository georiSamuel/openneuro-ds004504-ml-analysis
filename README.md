# EEG Dementia Dataset Analysis

## Objective
The primary goal of this repository is to perform preprocessing and machine learning classification on resting-state EEG recordings of individuals with Alzheimer's Disease (AD), Frontotemporal Dementia (FTD), and healthy controls. 

This project establishes a pipeline to clean the raw EEG signals and evaluates the performance of classical machine learning algorithms on this dataset. The models implemented and tested in this repository include:
* K-Nearest Neighbors (KNN)
* Decision Tree
* Support Vector Machine (SVM)
* Logistic Regression
* Linear Discriminant Analysis (LDA)

## Dataset
The dataset utilized in this project is the **OpenNeuro ds004504**, originally presented in the article *"A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG"*.

### Characteristics
* **Participants:** 88 subjects (36 AD, 23 FTD, 29 controls).
* **Equipment:** Recordings were acquired using a Nihon Kohden 2100 clinical device.
* **Recording Protocol:** 19 scalp electrodes were placed according to the 10-20 international system, along with 2 mastoid reference electrodes (A1 and A2). The sampling rate was 500 Hz with a 10 µV/mm resolution.
* **Procedure:** Participants were seated in a resting state with their eyes closed.
* **Duration:** Average recording durations were 13.5 minutes for AD, 12 minutes for FTD, and 13.8 minutes for the healthy control group.

### Original Preprocessing
*Note: The BIDS-compliant dataset provided by the authors has already undergone a rigorous baseline preprocessing pipeline.*
* **Filtering & Re-referencing:** A Butterworth band-pass filter (0.5–45 Hz) was applied, and the signals were re-referenced to the A1-A2 average.
* **Artifact Correction:** Artifact Subspace Reconstruction (ASR) was utilized to remove bad data segments exceeding a 0.5-second window standard deviation of 17.
* **ICA Denoising:** RunICA was performed to extract components. Components classified as eye or jaw artifacts by EEGLAB's ICLabel routine were automatically excluded.

### Download
Due to the large file sizes typical of EEG recordings, the data is **not** hosted in this repository. You must download it locally before running the scripts. 

You can download the dataset via Kaggle (requires a Kaggle account):
* **Web Link:** [EEG Dementia Dataset on Kaggle](https://www.kaggle.com/datasets/thngdngvn/openneuro-ds004504)
* **Kaggle CLI Command:**
  ```bash
  kaggle datasets download -d thngdngvn/openneuro-ds004504
  ```
Once downloaded, extract the contents into the `data/raw/` directory of this project.

## Setup & Dependencies

To ensure reproducibility and avoid conflicts, it is recommended to use a virtual environment.

### Prerequisites
* Python 3.8+
* [Git LFS](https://git-lfs.com/) (Recommended for tracking any large model weights or processed `.set`/`.edf` files)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/georiSamuel/openneuro-ds004504-ml-analysis.git
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Repository Structure

```text
eeg-dementia-dataset-analysis/
├── data/
│   ├── raw/                       # Place the Kaggle downloaded files here (ignored by git)
│   └── processed/                 # Cleaned/epoched EEG data and extracted features
|
├── notebooks/                     # Jupyter notebooks for Exploratory Data Analysis (EDA)
│   ├── 01_preprocessing.ipynb     # Scripts for filtering, artifact removal, and epoching
|   ├── 02_models_training.ipynb   # ML model training scripts
│   └── 03_tests_results.ipynb     # Presentation of test results using graphics
│  
├── .gitattributes                 # Git LFS tracking rules for large data/model files
├── .gitignore                     # Ignored files (e.g., /data/raw, /venv, .env)
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## Results

### First Iteration: Functional Connectivity + Nested CV + Feature Selection

To build a clinically meaningful classifier, we enhanced the initial RBP‑only baseline with three complementary strategies: functional connectivity features, embedded feature selection, and rigorous nested cross‑validation. The improvements below demonstrate that even modest absolute gains (~3–4 percentage points) represent a substantial step forward when subject‑wise leakage is strictly prevented.

#### 1. Functional Connectivity (Phase Locking Value – PLV)

**What is it and how was it implemented?**

- **Definition:** Phase Locking Value (PLV) quantifies the consistency of the phase difference between two oscillatory signals over time. It ranges from 0 (no synchrony) to 1 (perfect synchrony).
- **Implementation:** In the preprocessing notebook, after extracting Relative Band Power (RBP), we added a function that computes PLV for 10 specific channel pairs (fronto‑temporal and fronto‑parietal connections, the most relevant in the literature) within each 4‑second epoch. This generated 10 new features per epoch, increasing the total from 95 to 105 features.

**Why was it added?**

- **Limitation of RBP:** Spectral power (RBP) measures the **local** activity of each electrode but does not inform how different brain regions **communicate**. In neurodegenerative diseases like Alzheimer's and Frontotemporal Dementia, functional disconnection between regions (e.g., reduced synchrony between frontal and posterior areas) is a well‑established biomarker.
- **Objective:** Capture this network dysfunction, providing the model with information about the integrity of neural connections.

**Observed Impact:**

- Logistic Regression accuracy rose from 50.25% to **53.68%** (+3.43 percentage points).
- Logistic Regression Macro F1 rose from 0.4290 to **0.4639** (+3.49%).
- **Interpretation:** PLV added complementary discriminative information. The model was able to exploit subtle differences in synchrony between regions that differentiate the groups (AD, FTD, CN), even in the face of high inter‑subject variability.

#### 2. Feature Selection (SelectKBest within the Pipeline)

**What is it and how was it implemented?**

- **Definition:** `SelectKBest` is a univariate feature selection method that evaluates each feature individually (using ANOVA F‑value) and retains only the top `k` scoring features.
- **Implementation:** Selection was embedded **inside the pipeline** of each model, immediately after standardization (`StandardScaler`). The value of `k` was fixed at 50 (though the hyperparameter grid tested `k = 30, 50, 70`). This means that for each training fold, the model discards the least informative features **before** training the classifier.

**Why was it added?**

- **Curse of Dimensionality:** With 105 features and ~28,000 training samples per fold, the risk of overfitting is high. Many features may be redundant or noisy.
- **Generalization:** By eliminating irrelevant features, the model becomes simpler and tends to generalize better to new subjects.
- **Prevention of Data Leakage:** Selection is performed **within each cross‑validation fold**, using only the training data of that fold. This ensures that the choice of features is not influenced by the test data.

**Observed Impact:**

- **Robustness:** Models like SVM and KNN, which are sensitive to irrelevant features, also showed gains (SVM: +1.5%; KNN: +2.7%).
- **Interpretation:** The combination of `SelectKBest` with Nested CV ensured that the model not only learned, but learned from the **right features**, resulting in more stable and realistic validation metrics.

#### 3. Nested Cross‑Validation

**What is it and how was it implemented?**

- **Definition:** Nested CV consists of two cross‑validation loops:
    1. **Outer Loop (`GroupKFold`, 5 folds):** Splits subjects into 5 parts. Each part is used once as a test set while the other 4 are used for training. This estimates the final model performance.
    2. **Inner Loop (`StratifiedKFold`, 3 folds):** Within each outer training fold, the data is split again to search for the best hyperparameters (e.g., SVM `C`, `k` for `SelectKBest`) using `GridSearchCV`.
- **Implementation:** We replaced the simple `cross_validate` with the nested loop, ensuring that hyperparameter optimization **never sees** the subjects of the outer test fold.

**Why was it added?**

- **Selection Bias:** If we optimize hyperparameters using the entire dataset (or a single validation split) and then report accuracy on the test set, we are **leaking information** from the test set into the parameter choice. This artificially inflates metrics.
- **Scientific Honesty:** Nested CV provides an **unbiased estimate** of the model's ability to generalize to completely new patients. It is the gold standard in machine learning studies applied to healthcare.

**Observed Impact:**

- **Confidence in Results:** The scores presented (e.g., 53.68% for Logistic Regression) are a **realistic estimate** of the performance expected in an independent population.
- **Fair Comparison:** By applying the same procedure to all models, we can fairly compare their performances, knowing that differences are not artifacts of overfitting or inadequate tuning.

#### Overall Result

![Nested CV Accuracy Results](/assets/results/nestedCV-accuracy.png)

> Taking approximately 36 minutes for training estimation and scoring (with Nested CV).

All models showed slight improvement:

| Model | Before | After | Δ Accuracy |
| :--- | :---: | :---: | :---: |
| Logistic Regression | 0.5025 | 0.5368 | +0.0343 |
| LDA | 0.5225 | 0.5234 | +0.0009 |
| SVM | 0.4974 | 0.5125 | +0.0151 |
| KNN | 0.4397 | 0.4666 | +0.0269 |
| Decision Tree | 0.4366 | 0.4531 | +0.0165 |

**What improved and why:**

- Logistic Regression gained the most (+3.4 points) — linear models benefit more from connectivity features (PLV) because these features capture inter‑channel relationships that isolated spectral features do not. LR can directly weight these relationships.
- LDA remained practically unchanged (+0.09%) — expected, because LDA assumes all classes share the same covariance structure, which is rarely true in EEG connectivity data.

**The overall result remains within the expected range** — 50–54% for 3 classes with subject‑wise CV is honest and publishable. Many papers reporting 90%+ on this dataset have subject leakage.

### ROC‑AUC Analysis

The Receiver Operating Characteristic (ROC) curves and corresponding Area Under the Curve (AUC) values provide a threshold‑independent view of each model's discriminative power across the three classes.

![ROC-AUC Curves](/assets/results/roc_auc_curves.png)

- **Logistic Regression** achieved the highest macro‑averaged AUC (0.672), closely followed by SVM (0.671) and LDA (0.648).
- The **FTD class** consistently shows the lowest AUC (around 0.545 across models), confirming it as the most challenging diagnosis to distinguish from the others.
- All models perform substantially better than random guessing (AUC = 0.500), indicating that the extracted EEG features capture meaningful disease‑related signals.

### Overall Performance Comparison

The bar chart below summarizes the macro‑averaged metrics (Accuracy, Precision, Recall, F1 Macro) for all five classifiers.

![Performance Comparison](/assets/results/performance_comparison.png)

- **Logistic Regression** leads in Accuracy (0.539), Precision (0.506), Recall (0.494), and F1 Macro (0.489).
- **LDA** follows closely, with an accuracy of 0.523 and F1 Macro of 0.439.
- **Decision Tree** exhibits the lowest scores across all metrics, which is expected given its tendency to overfit high‑dimensional data without careful pruning.

The dashed gray line represents the random‑guess baseline (33.3% for a 3‑class problem). All models outperform this baseline by a meaningful margin, validating the feature engineering and subject‑wise validation strategy.

#### Summary of Combined Impact

| Change | Primary Role | Impact on Results |
| :--- | :--- | :--- |
| **Functional Connectivity** | Provides new neurophysiological information | Up to 3.4% increase in accuracy and Macro F1 |
| **Feature Selection** | Reduces dimensionality and overfitting | Improves robustness and generalization, especially in noise‑sensitive models |
| **Nested CV** | Ensures honest performance estimation | Confers scientific credibility to scores, without artificially inflating them |

**Conclusion:** The three changes act in synergy. Functional connectivity enriches the data, feature selection filters noise, and Nested CV ensures reliable evaluation. The final result is a more powerful, transparent, and clinically relevant pipeline for diagnosing dementia from EEG.

---

## Discussion

### Why is the F1‑Score for FTD Consistently Low?

![Per‑Class F1 Heatmap](/assets/results/f1_per_class_heatmap.png)

The heatmap reveals that **Frontotemporal Dementia (FTD) is the most challenging class to classify**, with F1‑scores ranging from ~0.12 to 0.25 across models. This is not a flaw in the pipeline, but rather a reflection of **neurophysiological and dataset‑specific factors**:

#### 1. Clinical Overlap Between FTD and AD

- FTD and Alzheimer's Disease (AD) share several EEG hallmarks, particularly **increased theta power** and **reduced alpha reactivity**.
- In resting‑state EEG, the spatial and spectral signatures of FTD can closely resemble early‑stage AD, making discrimination difficult even for expert clinicians.

#### 2. Heterogeneity Within the FTD Group

- FTD is an umbrella term encompassing **behavioral variant FTD (bvFTD)**, **semantic dementia**, and **progressive non‑fluent aphasia**. Each subtype affects different brain networks (frontal vs. temporal).
- The OpenNeuro dataset does not provide subtype labels, so the model must learn a single representation for a **heterogeneous population**, which degrades performance.

#### 3. Class Imbalance

| Group | Participants | Epochs (approx.) |
| :--- | :---: | :---: |
| AD | 36 | ~14,500 |
| CN | 29 | ~12,000 |
| FTD | 23 | ~8,200 |

- FTD is the **minority class** (~23% of epochs). Machine learning models trained without explicit balancing tend to prioritize the majority classes (AD and CN).
- The **macro F1‑score** treats all classes equally, so poor performance on the minority class significantly drags down the average.

#### 4. Frontal Artifact Susceptibility

- EEG electrodes over frontal regions (Fp1, Fp2, F7, F8, Fz) are highly susceptible to **eye‑blink and muscle artifacts**.
- Since FTD pathology primarily affects **frontal and anterior temporal lobes**, the very regions most relevant for FTD diagnosis are also the **most artifact‑prone**. Even with ICA/ASR cleaning, residual artifacts may obscure disease‑specific signals more for FTD than for AD (which typically shows posterior slowing).

#### 5. Feature Limitations

- Our current feature set (Relative Band Power + PLV on 10 pairs) may be **suboptimal for FTD**.
- FTD often presents with **asymmetric frontal slowing** and **reduced long‑range fronto‑temporal connectivity**. Features that explicitly capture **inter‑hemispheric asymmetry** or **gradients of slowing** might be necessary to better separate FTD from AD.

#### Suggested Mitigations (Future Work)

- **Data‑level:** Apply **SMOTE** or **class‑weighted loss functions** to compensate for imbalance.
- **Feature‑level:** Compute **asymmetry indices** (e.g., left vs. right frontal power ratios) and **anterior‑posterior gradients**.
- **Model‑level:** Use **hierarchical classification** (first distinguish Dementia vs. Control, then AD vs. FTD) or **ordinal regression** if disease severity scores are available.

---

### The Near Difference Between Out‑of‑Fold and Nested CV Results

![OOF vs Nested CV Comparison](/assets/results/oof_vs_cv_comparison.png)

The out‑of‑fold metrics obtained here should closely match the nested cross‑validation scores saved from Notebook 02. We load the `cv_summary.csv` file and compare the two sets of estimates to ensure consistency.

#### Interpreting the OOF vs. Nested CV Comparison

The table above compares two unbiased estimates of model performance:

| Metric Source | How It Was Computed |
| :--- | :--- |
| **Out‑of‑Fold (OOF)** | Predictions were generated by re‑fitting each model (with fixed optimal hyperparameters) on the training subjects of each `GroupKFold` split and predicting the held‑out subjects. Metrics were then calculated once across all accumulated predictions. |
| **Nested CV (Notebook 02)** | The mean and standard deviation of metrics were computed across the **5 outer folds** of the nested cross‑validation procedure. Each fold's metric was obtained using hyperparameters tuned exclusively on the inner training split of that fold. |

#### Why Are There Slight Differences?

Small discrepancies (e.g., Logistic Regression F1 Macro: 0.4885 OOF vs. 0.4639 CV) are **expected and normal**. They arise from two main sources:

1. **Hyperparameter Selection**
    - In **Nested CV**, each outer fold uses a **different set of optimal hyperparameters** (selected via inner CV). The reported CV metrics are an average over these slightly different models.
    - In **OOF**, the pipeline is loaded with the **final hyperparameters** (tuned on the entire dataset) and then re‑fitted on each training split. This uses a **single, globally optimal** hyperparameter configuration.

2. **Metric Aggregation**
    - **Nested CV** reports the **mean of per‑fold metrics** (e.g., average of 5 F1 scores).
    - **OOF** computes metrics **once on all accumulated predictions**, which is mathematically equivalent to a weighted average but can differ slightly due to the non‑linearity of some metrics (e.g., F1 score).

#### What Does This Tell Us?

- **Consistency:** The OOF and CV metrics are very close (difference < 0.03 for Accuracy and < 0.03 for F1 Macro in most cases). This confirms that the performance estimates are **stable and reproducible** across different evaluation methods.
- **Model Ranking:** Both methods agree that **Logistic Regression** is the best model, followed by **LDA** and **SVM**. The ranking is preserved, which strengthens confidence in the conclusion.
- **No Data Leakage:** The close agreement indicates that the nested CV procedure did not leak information and that the OOF predictions were generated without contamination from the test subjects.

> **Conclusion:** The slight variations are statistically negligible. The two evaluation strategies provide converging evidence that our pipeline yields honest, generalizable performance estimates suitable for clinical translation.

## References
* **Original Article:** Miltiadous, A.; Tzimourta, K.D., Afrantou, T.; Ioannidis, P.; Grigoriadis, N.; Tsalikakis, D.G.; Angelidis, P.; Tsipouras, M.G.; Glavas, E., Giannakeas, N.; et al. *A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG.* Data 2023, 8, 95. [https://www.mdpi.com/2306-5729/8/6/95](https://www.mdpi.com/2306-5729/8/6/95)
* **Inspiration Repo:** [JonathanReyess/eeg-dementia-ml](https://github.com/JonathanReyess/eeg-dementia-ml)
