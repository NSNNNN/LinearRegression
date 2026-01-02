import joblib
from data_preprocessing import load_data, preprocess_data
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = load_data("data/boston.csv")
X_train, X_test, y_train, y_test = preprocess_data(df)

model = joblib.load("model.pkl")
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R²:", r2_score(y_test, y_pred))
