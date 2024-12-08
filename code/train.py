from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

data_root = "../data/"
seed = 42

df = TabularDataset(data_root + "train_dataset.csv")
df = df.reset_index(drop=True)


def safe_division(numerator, denominator, default=0):
  """Safe division handling division by zero"""
  return np.where(denominator != 0, numerator / denominator, default)


def drop_columns(df):
  """Drop unnecessary columns"""
  drop_columns = [
    "ID",
    "r_generalcode1",
    "r_generalcode2",
    "r_generalcode4",
    "r_generalcode5",
  ]

  return df.drop(columns=drop_columns)


def handle_missing_values(df):
  """Handle missing values with appropriate strategies"""
  df = df.copy()

  # Simple zero imputation for income
  df["r_additional_income"] = df["r_additional_income"].fillna(0)
  df["r_spouse_income"] = df["r_spouse_income"].fillna(0)

  # Business type specific imputation for credit limit
  df["r_expected_credit_limit"] = df.groupby("c_business_type")["r_expected_credit_limit"].transform(
    lambda x: x.fillna(x.median())
  )

  # Mode imputation for categorical
  df["r_generalcode3"] = df["r_generalcode3"].fillna(4)
  df["r_propose"] = df["r_propose"].fillna(6)  # less than 1.5% missing fill other category
  df["c_date_of_salary_payment"] = df["c_date_of_salary_payment"].fillna(1)  # less than 1% missing
  df["Leasing enquiry_count"] = df["Leasing enquiry_count"].replace("*", 0)

  return df


def calculate_age(df):
  """Calculate age from date of birth"""
  df = df.copy()
  df["date_of_birth"] = pd.to_datetime(df["date_of_birth"])
  reference_date = pd.to_datetime("2024-01-01")
  df["age"] = (reference_date - df["date_of_birth"]).dt.days // 365

  df["pms_i_ymd"] = pd.to_datetime(df["pms_i_ymd"])
  # Create age at application
  df["age_application"] = (df["pms_i_ymd"] - df["date_of_birth"]).dt.days // 365

  return df.drop(["date_of_birth", "pms_i_ymd"], axis=1)


def calculate_income_features(df):
  """Calculate income-related features"""
  df = df.copy()

  # Basic income calculations
  df["total_income"] = df["c_monthly_salary"] + df["r_additional_income"]

  # Income ratios
  df["income_per_dependent"] = safe_division(df["total_income"], (df["number_of_children"] + 1))
  df["income_per_resident"] = safe_division(df["total_income"], df["number_of_resident"])

  # Drop original columns
  df = df.drop(columns=["c_monthly_salary", "r_additional_income"])

  return df


def calculate_stability_features(df):
  """Calculate stability-related features"""
  df = df.copy()

  # Employment stability
  df["employment_months"] = df["c_number_of_working_year"] * 12 + df["c_number_of_working_month"]
  df["job_stability_score"] = safe_division(df["employment_months"], df["age"], default=0).clip(max=1, min=0)

  # Living stability with minimum thresholds
  df["living_months"] = df["living_period_year"] * 12 + df["living_period_month"]
  df["residence_stability_score"] = safe_division(df["living_months"], df["age"], default=0).clip(max=1, min=0)

  # Drop original columns
  df = df.drop(
    columns=[
      "c_number_of_working_year",
      "c_number_of_working_month",
      "living_period_year",
      "living_period_month",
    ]
  )

  return df


def calculate_risk_features(df):
  """Calculate risk-related features"""
  df = df.copy()

  # Credit inquiries
  df["total_inquiries"] = df["Bank inquiry_count"] + df["Consumer finance inquiry_count"]
  df["high_inquiry_flag"] = (df["total_inquiries"] > 3).astype(int)

  # Debt metrics
  df["debt_burden"] = safe_division(df["r_allloan_amount"], (df["total_income"] * 12)).clip(min=0, max=10)
  df["credit_limit_ratio"] = safe_division(df["r_expected_credit_limit"], df["total_income"]).clip(min=0, max=10)

  # Business risk
  business_risk_map = {
    16: 1,
    14: 1,  # Low risk (government, education)
    15: 2,
    11: 2,
    12: 2,  # Medium-low risk (hospital, finance, insurance)
    4: 3,
    7: 3,  # Medium risk (manufacturing, business service)
    8: 4,
    5: 4,  # High risk (individual service, merchant)
  }
  df["business_risk_level"] = df["c_business_type"].map(business_risk_map).fillna(3)

  # Composite risk score
  df["high_risk_flags"] = (
    (df["living_months"] < 12).astype(int)
    + (df["employment_months"] < 12).astype(int)
    + (df["debt_burden"] > 0.5).astype(int)
    + (df["high_inquiry_flag"])
  )

  return df


def calculate_debt_profile(df):
  """Calculate comprehensive debt profile features"""
  df = df.copy()

  # Total debt exposure
  df["total_debt"] = (
    df["Overdraft_balance"]
    + df["Personal Loan_balance"]
    + df["Mortgage_balance"]
    + df["Credit Card_balance"]
    + df["Automobile installment purchase_balance"]
    + df["Other installment purchase_balance"]
    + df["Loan for agriculture_balance"]
    + df["Other Loans_balance"]
  )

  # Debt diversity score
  df["debt_products_count"] = (
    (df["Overdraft_count"] > 0).astype(int)
    + (df["Personal Loan_count"] > 0).astype(int)
    + (df["Mortgage_count"] > 0).astype(int)
    + (df["Credit Card_count"] > 0).astype(int)
    + (df["Automobile installment purchase_count"] > 0).astype(int)
  )

  # Average debt per product
  df["avg_debt_per_product"] = safe_division(df["total_debt"], df["debt_products_count"])

  return df


def calculate_purpose_features(df):
  """Calculate features related to loan purpose and intent"""
  df = df.copy()

  # Combine expected credit limit with purpose
  df["credit_purpose_ratio"] = safe_division(
    df["r_expected_credit_limit"], df["r_propose"].map(df.groupby("r_propose")["r_expected_credit_limit"].median())
  )

  return df


def define_data_type(df):
  """Define data types for each column"""
  df = df.copy()

  category_cols = [
    "Area",
    "Province",
    "postal_code",
    "r_propose",
    "Shop Name",
    "apply",
    "c_postal_code",
    "c_business_type",
    "c_employment_status",
    "c_occupation",
    "c_position",
    "c_salary_payment_methods",
    "c_date_of_salary_payment",
    "date_of_birth_week",
    "gender",
    "marital_status",
    "media",
    "r_generalcode1",
    "r_generalcode2",
    "r_generalcode3",
    "r_generalcode4",
    "r_generalcode5",
    "tel_category",
    "type_of_residence",
  ]
  for col in category_cols:
    if col in df.columns:
      df[col] = df[col].astype("category")

  int_cols = ["Leasing enquiry_count", "Bank inquiry_count", "Consumer finance inquiry_count"]

  for col in int_cols:
    if col in df.columns:
      if col == "Leasing enquiry_count":
        df[col] = df[col].replace("*", 0)
      df[col] = df[col].astype("int")

  return df


def preprocess(df):
  """Main preprocessing pipeline"""
  df = df.copy()

  # Basic preprocessing
  df = drop_columns(df)
  df = handle_missing_values(df)
  df = df.fillna(0)
  df = define_data_type(df)

  # Feature engineering
  df = calculate_age(df)
  df = calculate_income_features(df)
  df = calculate_stability_features(df)
  df = calculate_risk_features(df)
  df = calculate_debt_profile(df)
  df = calculate_purpose_features(df)

  # Encoding and scaling
  # df = encode_categorical_features(df)

  # df = df.replace([np.inf, -np.inf], np.nan)
  return df


data = preprocess(df)

data["y"] = data["default_12month"].astype("int")
data = data.drop(columns=["default_12month"])
data = data.reset_index(drop=True)

print(data.info())

train_data = data
train_data = train_data.reset_index(drop=True)

print(data["y"].value_counts(normalize=True))

test = TabularDataset(data_root + "public_dataset_without_gt.csv")
test = preprocess(test)

preset = [
  "best_quality",
  "high_quality",
  "good_quality",
  "medium_quality",
  "optimize_for_deployment",
][1]

# included = ["XGB", "GMB", "RF", "XT", "CAT"]
excluded_model_types = ["KNN", "NN_TORCH"]
# problem_type = "regression"
problem_type = "binary"


eval_metric = ["roc_auc", "f1", "average_precision"]

time_limit = 60 * 60
predictor = TabularPredictor(
  label="y",
  verbosity=2,
  problem_type=problem_type,
  # eval_metric=eval_metric[0],
  sample_weight="balance_weight",  # "auto_weight"
).fit(
  train_data=train_data,
  # test_data=test_data,
  presets=preset,
  time_limit=time_limit,
  num_gpus=1,
  # included_model_types=included,
  excluded_model_types=excluded_model_types,
  num_bag_folds=5,
  refit_full=True,
)
