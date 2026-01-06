import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier # يمكنك تغيير الموديل هنا
import joblib

df = pd.read_csv('parkinsons.csv')

X = df[['PPE', 'spread1']] 
y = df['status']


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

if accuracy >= 0.8:
    print("Success! Accuracy is above 0.8.")
    
    joblib.dump(model, 'my_model.joblib')
    print("Model saved as 'my_model.joblib'")
else:
    print("Accuracy is too low. Try choosing different features.")
