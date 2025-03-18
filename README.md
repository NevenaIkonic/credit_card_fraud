# Credit Card Fraud Detection

## 📌 Project Overview
This project focuses on detecting fraudulent transactions using machine learning techniques. The dataset consists of anonymized credit card transaction data, and the goal is to train a model that can effectively distinguish between fraudulent and legitimate transactions.

### 🔍 Key Features
- Data preprocessing and cleaning
- Handling JSON data for transaction processing
- Exploratory data analysis (EDA) with visualizations
- Implementation of a **Logistic Regression** classifier
- Handling class imbalance using **class weighting and sample weighting**
- Model evaluation and performance analysis 
- Testing on incoming transaction data 

---

## 📂 Project Structure
```
📁 project_root/
│-- data/                         # Folder containing dataset
│   ├── creditcard_modify.csv     # Processed transaction dataset
│-- plots/                        # Folder for saved visualizations
│-- src/                          # Source code directory
│   ├── preprocess.py             # Data preprocessing functions
│   ├── train.py                  # Model training script
│   ├── evaluate.py               # Model evaluation metrics
│-- README.md                     # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/NevenaIkonic/credit_card_fraud.git
cd credit-fraud-detection
```

### 2️⃣ Install Dependencies
Ensure you have Python 3.8+ and install the required libraries:
```bash
pip install -r requirements.txt
```

---

## 📊 Exploratory Data Analysis
Before training the model, we performed an in-depth analysis:
- 📉 **Distribution of transaction amounts**
- 🔥 **Correlation heatmaps between features**
- 📊 **Scatter plots for key variables**
- 🔄 **Class distribution visualization**

---

## 🤖 Model Training
The classifier used in this project is **Logistic Regression**. To handle the class imbalance, class and sample weighting techniques were applied.

---

## 📊 Refined Performance Evaluation
📏 **Evaluated classification metrics** to assess and improve model accuracy.

---

## ⚡ Real-Time Data Testing
✅ **Successfully tested the model on incoming transaction data** to validate fraud detection capabilities.

---

## 📈 Model Performance
To evaluate model effectiveness, we consider multiple metrics:

| Model Type   | Accuracy | Precision | Recall  | F1-score |
|-------------|----------|-----------|---------|-----------|
| **Unbalanced**  | **99.88%** | 69.05% | 58.39% | 63.27% |
| **Balanced**  | **96.41%** | 4.26% | **88.59%** | 8.13% |

### 🔍 Performance Analysis
- **Accuracy:** The unbalanced model achieves high accuracy (99.88%), but this is misleading due to the severe class imbalance—fraudulent transactions make up a tiny fraction of the dataset.  
- **Precision vs. Recall Tradeoff:**  
  - The unbalanced model has a **higher precision (69.05%)**, meaning it correctly identifies fraud more often when it predicts fraud. However, its **recall (58.39%) is low**, meaning it **misses** a significant number of actual fraud cases.  
  - The balanced model prioritizes **recall (88.59%)**, ensuring that most fraudulent transactions are caught, even at the cost of **lower precision (4.26%)**.  
- **Why Balance Matters?**  
  - In fraud detection, **missing fraud cases (false negatives) is far riskier** than falsely flagging a legitimate transaction (false positive).  
  - A balanced approach significantly improves recall, making the model more **reliable in real-world fraud prevention** despite its lower overall accuracy.  
  - The lower F1-score of the balanced model (8.13%) reflects the tradeoff, but in this context, **recall is the priority** for effective fraud detection.  

---

## 🙌 Acknowledgments
- Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

---

## 📩 Contact
For any questions or contributions, feel free to reach out via ikonicnena@gmail.com.
