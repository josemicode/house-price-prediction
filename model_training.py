import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def modelEvaluator(model, X_train, X_test, Y_train, Y_test):
    #Train the model:
    model.fit(X_train, Y_train)
    # Evaluation through prediction:
    Y_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    return mse

if __name__ == "__main__":
    # Fetch the split data:
    X_train, X_test, Y_train, Y_test = joblib.load('prep_data.pkl')
    # Model definition:
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=14) # num of trees, seed
    }