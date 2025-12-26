# Census Bureau Income Analysis & Customer Segmentation

This project contains comprehensive analysis of census bureau income data, including:
- **Task 1**: Income prediction using machine learning (classification)
- **Task 2**: Customer segmentation using K-Means clustering
- **EDA**: Exploratory data analysis supporting both tasks

## Quick Start

### 1. Activate Virtual Environment

```bash
source venv/bin/activate
```

### 2. Install Dependencies

**First time setup:**
```bash
pip install -r requirements.txt
```

**Verify installation:**
```bash
pip list | grep -E "pandas|scikit-learn|xgboost|lightgbm|umap"
```

### 3. Execute the Notebooks

**Option A: Run via Jupyter Notebook Interface**
```bash
jupyter notebook
```
Then navigate to the `notebooks/` folder and open the desired notebook.

**Option B: Run via Jupyter Lab**
```bash
jupyter lab
```

**Option C: Run All Cells Programmatically**
```bash
# Execute census income analysis (EDA)
jupyter nbconvert --to notebook --execute notebooks/census_income_analysis.ipynb

# Execute Task 1 (Income Prediction)
jupyter nbconvert --to notebook --execute notebooks/Assignment_task1.ipynb

# Execute Task 2 (Customer Segmentation)
jupyter nbconvert --to notebook --execute notebooks/Assignment_task2.ipynb
```

**Option D: Convert to Python Script and Run**
```bash
# Convert notebook to Python script
jupyter nbconvert --to python notebooks/Assignment_task1.ipynb

# Run the script
python notebooks/Assignment_task1.py
```

### 4. Deactivate Virtual Environment

When you're done:

```bash
deactivate
```

## Project Structure

```
JP Morgan Assignment/
├── requirements.txt                         # Python dependencies
├── notebooks/
│   ├── census_income_analysis.ipynb        # Exploratory Data Analysis (EDA)
│   ├── Assignment_task1.ipynb              # Task 1: Income Prediction Models
│   └── Assignment_task2.ipynb                         # Task 2: Customer Segmentation
├── models/                                       
├── TakeHomeProject/
│   ├── census-bureau.data                  # Raw census data (199,523 rows)
│   └── census-bureau.columns               # Column names/data dictionary
└── README.md                               # This file
```

## Notebooks Overview

### 1. census_income_analysis.ipynb (EDA)
**Purpose:** Comprehensive exploratory data analysis of census data  
**Runtime:** ~5-10 minutes  
**Outputs:** Visualizations, statistical tests, data quality reports  

### 2. Assignment_task1.ipynb (Income Prediction)
**Purpose:** Build machine learning models to predict income >$50K  
**Runtime:** ~15-25 minutes (includes hyperparameter tuning)  
**Outputs:** Trained models, performance metrics, feature importance plots  

**Key sections:**
1. **Data Preprocessing** (~2 min)
   - Handles missing values, encodes categoricals, scales numerics
   - Train-test split (70-30 stratified)
   - Creates 100+ features via target encoding

2. **Baseline Models** (~3 min)
   - Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM
   - Evaluation metrics: F1-Score, Precision, Recall, AUC-PR
   - Best baseline: LightGBM F1 = 0.4634

3. **Feature Importance Analysis** (~2 min)
   - Identifies top predictors: occupation, age, education
   - Validates EDA findings
   - Generates visualizations for interpretation

4. **Model Enhancement Experiments** (~5 min)
   - Experiment 1: SMOTE for class imbalance
   - Experiment 2: Threshold tuning (precision vs recall)
   - Experiment 3: Sample weights for population representation

5. **Final Model Training** (~10 min)
   - GridSearchCV hyperparameter tuning (243 combinations)
   - 5-fold cross-validation
   - Final model: LightGBM F1 = 0.5123 (+10.6% over baseline)

6. **Model Interpretation** (~3 min)
   - Feature importance (gain vs split)
   - Confusion matrix analysis
   - Business recommendations

**Model Outputs:**
- Trained LightGBM classifier (pickled)
- Performance metrics CSV
- Feature importance rankings
- Threshold optimization results

### 3. Assignment_task2.ipynb (Customer Segmentation)
**Purpose:** Create demographic/economic customer segments using K-Means  
**Runtime:** ~10-15 minutes (includes dimensionality reduction)  
**Outputs:** Customer segments CSV, segment profiles, marketing recommendations  

**Key sections:**
1. **Feature Selection** (~1 min)
   - 19 demographic + economic features
   - Excludes income labels (unsupervised task)

2. **Data Preprocessing** (~3 min)
   - Handles missing values and "Not in universe"
   - Engineers financial features (log transforms, binary indicators)
   - One-hot + frequency encoding by cardinality
   - StandardScaler normalization

3. **K-Means Clustering** (~5 min)
   - Tests K=3 to K=8 clusters
   - Evaluates: Inertia, Silhouette, Davies-Bouldin, Calinski-Harabasz
   - Stability testing (ARI, NMI across random seeds)
   - Optimal K=6 selected

4. **Segment Refinement** (~2 min)
   - Identifies micro-segments (<3% population)
   - Merges into nearest large segments
   - Final: 5 actionable segments

5. **Segment Profiling** (~3 min)
   - Weighted statistics (using survey weights)
   - Demographic profiles (age, education, marital status)
   - Economic profiles (occupation, industry, employment)
   - Income characteristics (for interpretation only)

6. **Marketing Segment Cards** (~2 min)
   - Segment 0: Affluent Asset-Income (3.6%, premium targeting)
   - Segment 1: Children/Dependents (27.2%, family marketing)
   - Segment 2: Mainstream Working Households (42.3%, core market)
   - Segment 4: Older/Retirement-Age (21.2%, value-focused)
   - Segment 5: Union-Connected Workers (5.6%, paycheck-timed)

## Execution Order (Recommended)

1. **First:** `census_income_analysis.ipynb` (understand the data)
2. **Second:** `Assignment_task1.ipynb` (supervised learning)
3. **Third:** `Assignment_task2.ipynb` (unsupervised learning)

**Note:** Task 1 and Task 2 are independent and can run in parallel.

