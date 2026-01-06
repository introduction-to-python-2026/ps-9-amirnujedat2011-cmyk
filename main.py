import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

def evaluate():
    df = pd.read_csv('parkinsons.csv')
    features = ['PPE', 'spread1']
    X = df[features]
    y = df['status']
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    try:
        model = joblib.load('my_model.joblib')
        accuracy = model.score(X_scaled, y)
        print(f"Accuracy: {accuracy}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    evaluate()
