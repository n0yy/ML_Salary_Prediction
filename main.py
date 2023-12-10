import pandas as pd
from utils.preprocessor import preprocessor
from utils.predict import predict
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost.data")

# Get Data
cols = ["Age", "Gender", "Education Level", "Job Title", "Years of Experience"]

X = [
    [23, "Male", "Bachelor's", "Data Analyst", 2],
    [30, "Female", "PhD", "Software Engineer", 7],
    [20, "Male", "Bachelor's", "Junior Web Developer", 1],
    [35, "Male", "Master's", "Director of Marketing", 7],
]

data = pd.DataFrame(X, columns=cols)

# Preprocess Data
df = preprocessor(data)

# Predicting
data["Estimated Salary"] = predict(df)

print(data)