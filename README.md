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
   git clone [https://github.com/your-username/eeg-dementia-dataset-analysis.git](https://github.com/your-username/eeg-dementia-dataset-analysis.git)
   cd eeg-dementia-dataset-analysis
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

## References
* **Original Article:** Miltiadous, A.; Tzimourta, K.D., Afrantou, T.; Ioannidis, P.; Grigoriadis, N.; Tsalikakis, D.G.; Angelidis, P.; Tsipouras, M.G.; Glavas, E., Giannakeas, N.; et al. *A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG.* Data 2023, 8, 95. [https://www.mdpi.com/2306-5729/8/6/95](https://www.mdpi.com/2306-5729/8/6/95)
* **Inspiration Repo:** [JonathanReyess/eeg-dementia-ml](https://github.com/JonathanReyess/eeg-dementia-ml)
