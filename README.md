# Exploring the potential of ML/DL for Alzheimer's detection  

# Study characteristics

## **Abstract**
Alzheimer's disease (AD) is a Neurodegenerative disorder that affects millions of individuals around the globe. Early diagnosis of Alzheimer's disease is essential for prompt intervention and enhanced patient outcomes. In this study, the efficacy of various machine learning (ML) models for AD detection is investigated. We utilized linear models such as logistic regression, tree-based models such as random forest classifier (RFC) and decision tree classifier (DTC), probabilistic models, non-linear models such as support vector machines (SVM), and an ensemble model that combines the best of all the above models. In addition to traditional machine learning methods, we deployed a neural network model based on Few Shot Learning. Few-shot learning is distinguished by its capacity to learn new concepts or tasks with limited training data. It simulates human-like learning, in which humans can rapidly comprehend new information or identify new objects with only a few examples. Few-shot learning paves the way for more efficient and adaptable artificial intelligence systems that can quickly adapt to new environments or changing user needs especially in medical imaging, in which it can aid in the detection of rare diseases where labeled samples are scarce.

---

## **Dataset Description** 
For Alzheimer's Detection using EEG. The dataset is downloaded from https://osf.io/s74qf/ .The database was created by Florida State University researchers using a Biologic Systems Brain Atlas III Plus workstation to record from the 19 scalp (Fp1, Fp2, Fz, F3, F4, F7, F8, Cz, C3, C4, T3, T4, Pz, P3, P4, T5, T6, O1, and O2) loci of the worldwide 10-20 system. Cerebral lobes F–T (F: frontal, C: central, P: parietal, O: occipital, and T: temporal). Four groups—A, B, C, and D—recorded in two rest states: eyes open (A and C) by visually fixating and eyes closed (B and D) utilizing a linked-mandible reference forehead ground. 24 healthy seniors comprise Groups A and B. Groups C and D have 24 likely AD patients . 8-s EEG segments band-limited to 1-30 Hz were recorded at 128 Hz and extracted without eye motion, blinking. EEG technicians tracked each patient's attention.The dataset sample size of 48 is converted to 192 in total using Data Augmentation techniques such as  
  
  - **Time Shifting & Noising**: Time shifting involves shifting the entire time series forward or backward by a certain number of time steps. This technique helps introduce variations in the temporal structure of the data. By shifting the time series, you can simulate different time lags, delays, or phase shifts, which can be valuable for modeling time-dependent patterns or capturing temporal dependencies.Noising is the process of adding random noise to the time series data. This technique introduces randomness and perturbations into the data, which can help improve the robustness of models and reduce overfitting. Various types of noise can be added, such as Gaussian noise, random walk noise, or seasonal noise, depending on the characteristics of the data and the desired augmentation effect.

- **Rolling Mean**: It calculates the average of a specific window of time steps across the time series. The window size determines the number of adjacent data points considered for each average calculation. By taking the rolling mean, the time series is smoothed, and high-frequency noise or fluctuations are attenuated.

---

## **Model**
The models used for the EEG dataset (Augmented) are tree-based models, probability-based models, and non-linear models. Combined predictions from the best models are used to generate the final output using majority voting.

On the 48 samples of the EEG dataset (Non-Augmented), Few Shot Learning neural network models have been deployed. A substantial quantity of labeled data is required for deep learning models to generalize effectively and perform accurately. In medical imaging, it may be impractical or impossible to acquire such vast quantities of labeled data for each class. The concept of "few-shot learning" is now relevant. Few-shot learning is a subset of machine learning that enables models to learn from a small number of labeled examples, thereby overcoming the issue of insufficient data. The few-shot classification method employs a Siamese Network to learn embeddings based on sample similarities. The network requires a Two Input of arbitrary class. The network creates embeddings for both similar (same class pair) and dissimilar (different class pair) pairs during training. Embeddings of the pair are taught to be near for pairs of the same class and far for pairs of different classes. During the prediction process, the network compares the embeddings of the new sample to those of the reference samples of different classes by calculating similarity or dissimilarity scores (comparison score) between them using Euclidean distance or cosine similarity, and labels the new sample accordingly. For example, sample 'x' will be assigned to class 'A' if its embeddings resemble those of class 'A' samples more than those of reference samples from other classes. So, Instead of classifying whether a patient's sample is indicative of Alzheimer's disease or not, we may simply compare it to the Alzheimer's reference sample and the Non-Alzheimer's reference sample (healthy sample) and assign it to the closest class. Consequently, the network can classify new samples even when labeled data are scarce. 

---

## **Results**  

**Machine Learning (Augmented Dataset)** : 
<br>  

Performance scores of chosen models on different subsets of training data  

[![Different varieties of Machine Learning models are trained, and the best among them is selected and implemented as a combined classifier.](results/results_ml_bestmodels_for_combined_classifer.jpg)](results/results_ml_bestmodels_for_combined_classifer.jpg)  

Performance scores of all the models  

[![Performance scores](results/results_ml_models.jpg)](results/results_ml_models.jpg)  
<br>
**Few Shot Learning (Non - Augmented Dataset)** :  

Architecture of Few Shot Siamese Network  

![Alt Text](results/fewshot_model_arch.jpg)  

Siamese Network Dissimilarity Scores  

![Alt Text](results/results_fewshot.jpg)  

---

# Respository Usage Guide


## File tree

```
alzheimers-detection-methodologies-organized
┃
┣ 📜 README.md (you are here!)
┃
┣ 📦 data
┃ ┣ raw
┃ ┃ ┣ dataset_non_augmented.zip
┃ ┃ ┗ DataBase 
┃ ┃   ┣ SETA
┃ ┃   ┣ SETB
┃ ┃   ┣ SETC
┃ ┃   ┗ SETD
┃ ┗ processed
┃   ┣ train_augmented.csv
┃   ┗ test_augmented.csv
┃
┣ 📓 notebooks
┃ ┣ few_shot_similarity_nn
┃ ┃ ┣ 01_FewShotLearning_Similaritybased_DataPreprocessing.ipynb
┃ ┃ ┣ 02_FewShotLearning_Similarity_Model_train.ipynb
┃ ┃ ┗ 03_FewShotLearning_Similaity_testing.ipynb
┃ ┗ ml_based_models
┃   ┣ 01_AlzheimerDetection_Data_Preprocessing.ipynb
┃   ┗ 02_AlzheimersDetection_ML_Models.ipynb
┃
┣ 🤖 model
┃ ┣ model.py
┃ ┗ weights_for_fewshot
┃
┗ 🖼️ results
  ┣ fewshot_model_arch.jpg
  ┣ results_fewshot.jpg
  ┣ results_ml_bestmodels_for_combined_classifer.jpg
  ┗ results_ml_models.jpg
```

---

## Structure Explanation

1. **`data/`**: The core data directory of the project. It is now divided into two subdirectories to separate original and modified files, ensuring data integrity.
   - **`raw/`**: Contains the original, dataset. It includes the compressed [`dataset_non_augmented.zip`](data/raw/dataset_non_augmented.zip) and the `DataBase` folder (the extraction of the .zip). The EEG (Electroencephalogram) data is divided into 4 specific sets:
     - **SETA**: Healthy patients with open eyes.
     - **SETB**: Healthy patients with closed eyes.
     - **SETC**: Alzheimer's patients with open eyes.
     - **SETD**: Alzheimer's patients with closed eyes.
   - **`processed/`**: Stores the data after it has been cleaned, transformed, and artificially augmented. Includes the final tabular data used to train and evaluate the models ([`train_augmented.csv`](data/processed/train_augmented.csv) and [`test_augmented.csv`](data/processed/test_augmented.csv)).

2. **`notebooks/`**: Contains all the Jupyter Notebooks used for exploration, preprocessing, and execution. The files are numbered to indicate the exact execution order:
   - [**`few_shot_similarity_nn/`**](notebooks/few_shot_similarity_nn): Focuses on the Few-Shot Learning approach using Deep Neural Networks. It guides the user step-by-step through data preprocessing (01), model training (02), and final similarity-based testing (03).
   - [**`ml_based_models/`**](notebooks/ml_based_models): Dedicated to traditional Machine Learning algorithms (like Random Forest, SVM, etc.). Contains notebooks to prepare the tabular data (01) and to train/evaluate these classical models (02).

3. **`model/`**: A dedicated folder that isolates the core deep learning architecture (Few Shot Learning) from the experimental notebooks, making the code more modular and reusable.
   - [**`model.py`**](model/model.py): The raw Python script that defines the underlying architecture of the Neural Network (using a Siamese Network approach to calculate similarity).
   - [**`weights_for_fewshot`**](model/weights_for_fewshot) ⚠️: Contains a reference/link to the pre-trained weights of the model. This allows new users to load the "knowledge" of the network directly into the testing notebook without needing to retrain it from scratch.

4. **`results/` (Outputs and Visualizations)**: A folder dedicated to storing the visual outputs and metrics generated during the experiments. It includes architectural diagrams of the neural network ([`fewshot_model_arch.jpg`](results/fewshot_model_arch.jpg)) and charts comparing the performance, accuracy, and confusion matrices of the different tested approaches.


---

## Workflow

```mermaid
graph TD
    A[Step 1: Raw Data Preparation] --> B{Step 2: Choose Methodology}
    
    B -->|Path A: Few-Shot NN| C1[Preprocessing]
    C1 --> C2[Train or Load Model]
    C2 --> C3[Evaluation]
    
    B -->|Path B: Classical ML| D1[Feature Extraction]
    D1 --> D2[Train & Evaluate Models]
    
    C3 --> E[Step 3: Save Results]
    D2 --> E
```

### Step 1: Acquisition and Preparation of Raw Data
1. The user clones the repository to their local machine or Google Colab.
2. Access the `data/raw/` folder and unzip the `dataset_non_augmented.zip` file.
3. Explore the `database/` folder to understand the original electroencephalogram (EEG) signals separated by the 4 sets (SETA to SETD).

### Step 2: Choosing the Methodology
The project is divided into two distinct approaches. The user must choose which path to follow (or test both for comparison).


>Path A: Neural Networks Approach (Few Shot Learning model)

If the goal is to use similarity-based Artificial Intelligence:

* **Preprocessing:** Open and run the notebook `notebooks/few_shot_similarity_nn/01_FewShotLearning_Similaritybased_DataPreprocessing.ipynb`. This script will read the original data, create similarity pairs, and save the processed data in the `data/processed/` folder.
* **Training or Loading:** * *Option 1 (Train from scratch):* Run the `02_FewShotLearning_Similarity_Model_train.ipynb` notebook. The code will import the architecture from the `model/model.py` file, train the network with the processed data, and generate new weights.
    * *Option 2 (Use the pre-trained model):* The user opens the `model/weights_for_fewshot` file, accesses the Google Drive link, downloads the pre-trained weights, and saves hours of computation.
* **Evaluation:** Run the `03_FewShotLearning_Similaity_testing.ipynb` notebook, applying the model to the test data (which the AI has never seen) to obtain the final metrics.


>Path B: Classical Machine Learning Approach

If the goal is to use traditional statistical algorithms (such as Random Forest or SVM):

* **Phase 1 (Feature Extraction):** Run the notebook `notebooks/ml_based_models/01_AlzheimerDetection_Data_Preprocessing.ipynb`. This script cleans the raw data and uses techniques like PCA (Principal Component Analysis) to extract essential features, saving the resulting tabular files in the `data/processed/` folder.
* **Phase 2 (Training and Multiple Evaluation):** Run the `02_AlzheimersDetection_ML_Models.ipynb` notebook. The script will load the processed CSV files, simultaneously train several classical Machine Learning models, and present the confusion matrices and accuracy of each at the end.



### Step 3: Analysis and Recording of Results
Regardless of the chosen path, the final step is to compile the charts generated by the notebooks (such as the architecture diagram, performance charts, and comparative tables) and save them in the `results/` folder. 
