# Fraud Detection in Financial Transactions

A Python-based Machine Learning project for detecting potentially fraudulent financial transactions using supervised classification techniques.

The project analyzes transaction data, performs data preprocessing, trains multiple machine learning models, compares their performance, and evaluates fraud predictions using classification metrics.

---

## 📌 Project Overview

Financial fraud detection is an important application of Machine Learning in the financial sector. Large numbers of transactions need to be analyzed to identify suspicious or fraudulent activities.

This project develops a fraud detection system using transaction-related information such as transaction type, transaction amount, account balances, and transaction identifiers.

The system uses three Machine Learning classification algorithms:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier

The models are trained using historical transaction data where the `isFraud` column represents the target variable.

---

## 🎯 Objectives

The main objectives of this project are:

- Analyze financial transaction data.
- Identify patterns associated with fraudulent transactions.
- Preprocess categorical and numerical transaction features.
- Handle missing values when necessary.
- Split the dataset into training and testing data.
- Train multiple Machine Learning classification models.
- Compare model accuracy.
- Evaluate the Random Forest model using a confusion matrix and classification report.
- Predict whether a new financial transaction is fraudulent or non-fraudulent.
- Visualize transaction and fraud distributions.

---

## ✨ Features

### Data Analysis

- Dataset loading and validation
- Dataset shape inspection
- Column verification
- Missing-value analysis
- Descriptive statistical analysis
- Display of sample transaction records

### Data Preprocessing

The project handles both categorical and numerical transaction features.

Categorical data is processed using:

- Missing-value imputation
- Ordinal encoding
- Unknown-category handling

Numerical data is processed using:

- Missing-value imputation

The preprocessing is integrated into the Machine Learning pipeline so that the same fitted transformations are applied consistently to training data, testing data, and new transactions.

---

## 🤖 Machine Learning Models

The project compares three classification algorithms.

### 1. Logistic Regression

Logistic Regression is used as a baseline classification model for distinguishing fraudulent and non-fraudulent transactions.

### 2. Support Vector Machine

Support Vector Machine (SVM) is used to identify a decision boundary between fraudulent and non-fraudulent transactions.

### 3. Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve classification performance.

The Random Forest implementation uses:

- `200` decision trees
- `class_weight="balanced"`
- `random_state=2529`

---

## 📊 Model Evaluation

The project compares the accuracy of:

| Model | Evaluation |
|---|---|
| Logistic Regression | Accuracy |
| Support Vector Machine | Accuracy |
| Random Forest Classifier | Accuracy |

The actual accuracy values are calculated when the program is executed on the `Fraud.csv` dataset.

The project also evaluates the Random Forest model using:

- Confusion Matrix
- Classification Report
- Precision
- Recall
- F1-score

---

## 📈 Data Visualization

The project generates visualizations for:

### Transaction Type Distribution

Shows the distribution of different transaction types in the dataset.

### Fraudulent vs Non-Fraudulent Transactions

Shows the proportion of fraudulent and non-fraudulent transactions.

### Model Accuracy Comparison

Compares the accuracy achieved by:

- Logistic Regression
- Support Vector Machine
- Random Forest

---

## 📂 Dataset

The project uses a CSV file named:

```text
Fraud.csv
```

The database contains financial transaction information.

Required Database Columns
| Column |	Description |
|---|---|
| step | Transaction time step |
| type |	Type of financial transaction |
| amount | Transaction amount |
| nameOrig	| Originating account identifier |
| oldbalanceOrg	| Original balance of the originating account |
| newbalanceOrig	| New balance of the originating account |
| nameDest	| Destination account identifier |
| oldbalanceDest	| Original balance of the destination account |
| newbalanceDest	| New balance of the destination account |
| isFraud	| Fraud classification target |
| isFlaggedFraud	| Fraud flag indicator |

---

## 🎯 Target Variable

The primary target variable is:
```text
isFraud
```

The target represents whether a transaction is fraudulent.
```text
0 = Non-Fraudulent
1 = Fraudulent
```
The Machine Learning models use the transaction features to predict this target.

---

## 🔄 Project Workflow

                Fraud.csv
                    |
                    v
            Load Dataset
                    |
                    v
          Validate Database
                    |
                    v
          Data Inspection
                    |
                    v
       Missing Value Handling
                    |
                    v
        Feature Preprocessing
                    |
          +---------+---------+
          |                   |
          v                   v
     Categorical          Numerical
      Features             Features
          |                   |
          v                   v
       Encoding          Imputation
          |                   |
          +---------+---------+
                    |
                    v
           Train/Test Split
                    |
          +---------+---------+
          |         |         |
          v         v         v
      Logistic     SVM    Random Forest
     Regression
          |         |         |
          +---------+---------+
                    |
                    v
          Model Evaluation
                    |
                    v
       Accuracy Comparison
                    |
                    v
       Random Forest Analysis
                    |
          +---------+---------+
          |                   |
          v                   v
   Confusion Matrix    Classification Report
                    |
                    v
          New Transaction
                    |
                    v
             Prediction
                    |
          +---------+---------+
          |                   |
          v                   v
       Fraudulent       Non-Fraudulent


---

## 📁 Project Structure

```text
fraud-detection/
│
├── fraud_detection.py
├── Fraud.csv
├── requirements.txt
├── README.md
│
└── outputs/
    └── generated visualizations
```
---


## 🛠️ Technologies Used

| Technology          | Purpose                       |
| ------------------- | ----------------------------- |
| Python              | Main programming language     |
| Pandas              | Data loading and manipulation |
| NumPy               | Numerical operations          |
| Matplotlib          | Data visualization            |
| Scikit-learn        | Machine Learning              |
| Logistic Regression | Classification                |
| SVM                 | Classification                |
| Random Forest       | Classification                |

---

## 📦 Python Libraries

Install the required packages using:
```text
pip install -r requirements.txt
```

The main dependencies are:

```text
pandas
numpy
matplotlib
scikit-learn
```

---

## ⚙️ Installation

### Step 1: Clone the Repository
```text
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

### Step 2: Open the Project

Open the project folder in Visual Studio Code.

### Step 3: Create a Virtual Environment

```text
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

For Windows:

```text
.venv\Scripts\activate
```

### Step 5: Install Dependencies

```text
pip install -r requirements.txt
```
### Step 6: Add the Database

Place the original:

```text
Fraud.csv
```
in the same directory as:
```text
fraud_detection.py
```

---

## ▶️ How to Run

Execute the following command in the VS Code terminal:

```text
python fraud_detection.py
```
---

## 🔍 Preprocessing Pipeline

The project automatically identifies categorical and numerical features.

Categorical Features

Examples include:

```text
type
nameOrig
nameDest
```

These features are processed using ordinal encoding.

Unknown categories are handled using:

```text
unknown_value = -1
```

Numerical Features

Examples include:

```text
step
amount
oldbalanceOrg
newbalanceOrig
oldbalanceDest
newbalanceDest
```

Missing numerical values are handled using median imputation.

---

## ⚖️ Class Imbalance Handling

Fraud datasets commonly contain significantly fewer fraudulent transactions than legitimate transactions.

To help address class imbalance, the classification models use:

```text
class_weight="balanced"
```

This is configured for Logistic Regression, SVM, and Random Forest.

---

## 🧪 Train/Test Split

The dataset is divided into training and testing subsets.

The implementation uses:

```text
Training data: 80%
Testing data: 20%
Random state: 2529
```

Where possible, stratification is used to preserve the distribution of the target classes between the training and testing datasets.

---

## 📊 Evaluation Metrics
### Accuracy

Measures the overall proportion of correctly classified transactions.

### Confusion Matrix

The confusion matrix shows:

```text
True Negative
False Positive
False Negative
True Positive
```

### Precision

Measures how many transactions predicted as fraudulent were actually fraudulent.

### Recall

Measures how many actual fraudulent transactions were correctly detected.

### F1-Score

Provides a combined measure of precision and recall.
