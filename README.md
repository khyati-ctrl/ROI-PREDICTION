# ROI Prediction Model

A machine learning project that estimates the financial benefits and Return on Investment (ROI) associated with corporate AI adoption. The project compares XGBoost and Random Forest regression models to identify the better-performing model for predicting financial benefits.

## Overview

As organizations increasingly invest in Artificial Intelligence, estimating the potential financial returns of these investments can support better decision-making.

This project uses corporate AI adoption data to predict the total financial benefit associated with an organization's AI investment. The predicted benefit is then used to calculate an estimated ROI percentage.

The project covers data preprocessing, feature engineering, outlier handling, categorical encoding, regression model comparison, performance evaluation, and model serialization for downstream application integration.

## Objectives

- Predict financial benefits associated with corporate AI adoption.
- Explore and preprocess financial and organizational data.
- Compare XGBoost and Random Forest regression models.
- Evaluate model performance using R² score and Mean Absolute Error (MAE).
- Select the better-performing model based on test-set R².
- Save the selected model for use in a prediction pipeline.

## Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Model Evaluation:** R² Score, Mean Absolute Error (MAE)
- **Model Serialization:** Pickle
- **Development Environment:** Jupyter Notebook

## Dataset

The project uses a corporate AI adoption dataset containing organizational, financial, and AI implementation features.

Key input features include:

- AI investment in USD
- Employee AI training hours
- Number of AI deployments
- AI adoption level
- Automation rate
- Productivity gain
- AI maturity score
- Industry
- Country
- Year

The target variable is **total financial benefit**, calculated as:

`Total Financial Benefit = Cost Savings + Revenue Impact`

The ROI percentage is calculated separately using the predicted financial benefit and the initial AI investment:

`ROI (%) = (Predicted Financial Benefit / AI Investment) × 100`

For non-positive investments, the current implementation returns an ROI of 0%.

## Methodology

### 1. Data Exploration and Feature Engineering

- Loaded the corporate AI adoption dataset using Pandas.
- Created a total financial benefit target by combining cost savings and revenue impact.
- Calculated ROI percentages for exploratory analysis.
- Examined numerical features for potential outliers using the Interquartile Range (IQR) method.

### 2. Outlier Handling

Applied 1st and 99th percentile Winsorization to the following columns:

- AI investment
- Employee AI training hours
- Total financial benefit

This caps extreme values at the selected percentile boundaries rather than removing entire observations.

### 3. Feature Preparation

- Removed the company identifier and target-derived financial columns from the input features.
- Applied one-hot encoding to categorical industry and country features.
- Prepared a feature matrix containing 31 input columns.
- Split the data into training and testing sets using an 80:20 ratio and `random_state=42`.

The resulting split contained 160,000 training observations and 40,000 test observations.

### 4. Model Training

Two regression algorithms were trained and evaluated on the same train-test split.

**XGBoost Regressor**
- Number of estimators: 150
- Learning rate: 0.08
- Maximum depth: 6
- Random state: 42

**Random Forest Regressor**
- Number of estimators: 100
- Maximum depth: 12
- Random state: 42

Both models used parallel processing through `n_jobs=-1`.

### 5. Model Evaluation and Selection

The models were evaluated using two metrics:

- **R² Score:** Measures how much of the variation in the target is explained by the model. Higher is better.
- **Mean Absolute Error (MAE):** Measures the average absolute difference between predicted and actual financial benefits. Lower is better.

The model with the higher test-set R² score was selected as the winning model.

## Model Comparison

The following results were obtained from the notebook's test-set evaluation:

| Metric | XGBoost | Random Forest |
|---|---:|---:|
| R² Score | 0.84421 | 0.84270 |
| MAE (USD) | $1,115,139.12 | $1,120,982.93 |

### Why XGBoost Was Selected

XGBoost achieved the higher R² score and the lower MAE among the two evaluated models.

- Its R² score was approximately 0.844, indicating that it explained about 84.4% of the variation in the test-set target.
- Its MAE was approximately $1.115 million.
- Random Forest achieved an R² score of approximately 0.843 and an MAE of approximately $1.121 million.

Based on these results, **XGBoost was selected as the best-performing model among the two candidates tested**.

The performance difference is relatively small, so the results establish a comparative advantage on this test split rather than proving that XGBoost will outperform Random Forest on every dataset.

## Prediction Pipeline

After model evaluation, the selected model was serialized to a Pickle file named `champion_roi_model.pkl`.

The prediction pipeline accepts organizational and AI adoption inputs, prepares a feature matrix, predicts the financial benefit, and calculates the estimated ROI percentage.

The returned prediction contains:

- `predicted_financial_benefit_usd`: Predicted financial benefit in USD.
- `roi_percentage`: Estimated ROI percentage calculated from the predicted benefit and initial AI investment.

An example test input in the notebook uses an AI investment of $5,000,000 and produces the following output:

```python
{
    "predicted_financial_benefit_usd": 2900437.0,
    "roi_percentage": 58.01
}
```

This is an illustrative prediction from the notebook's test pipeline, not a guarantee of actual business returns.

## Project Structure

```text
ai-roi-prediction/
├── final-roi-prediction.ipynb
├── corporate_ai_adoption_dataset.csv
├── champion_roi_model.pkl
├── requirements.txt
└── README.md
```

The dataset and trained model files should be included only if you intend to distribute them and have permission to do so. The Pickle model file is generated after running the training and evaluation notebook.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/ai-roi-prediction.git
cd ai-roi-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `final-roi-prediction.ipynb` and execute the cells in order to explore the data, train the models, compare their performance, and generate the serialized model artifact.

The prediction pipeline requires the trained model file to be present in the expected working directory.

## Future Improvements

- Tune hyperparameters using cross-validation and systematic search.
- Evaluate additional regression algorithms and compare them on the same splits.
- Add RMSE and residual analysis to assess prediction errors.
- Investigate prediction uncertainty and performance across different industries and investment ranges.
- Package preprocessing and prediction into a reusable Scikit-learn pipeline.
- Build a FastAPI endpoint or web interface for interactive predictions.
- Validate the feature engineering and evaluation pipeline on independent data.

## Limitations

- Model performance depends on the quality and representativeness of the dataset.
- Evaluation results are based on a single 80:20 train-test split.
- MAE is expressed in USD, and errors can be substantial in absolute monetary terms.
- The ROI percentage is derived from predicted financial benefit; it is not a separately trained ROI target.
- Predictions are estimates based on the dataset and should not be interpreted as guaranteed financial outcomes.

## Author

**Khyati Sharma**

B.Tech Computer Science Engineering | AI & Machine Learning

GitHub: [khyati-ctrl](https://github.com/khyati-ctrl)

---

*Built as a machine learning project exploring financial benefit prediction, regression model comparison, and ROI estimation for corporate AI adoption.*
