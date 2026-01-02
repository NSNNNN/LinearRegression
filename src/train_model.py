from data_preprocessing import load_data, preprocess_data
from sklearn.linear_model import LinearRegression
import joblib

df = load_data("data/boston.csv")
X_train, X_test, y_train, y_test = preprocess_data(df)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
