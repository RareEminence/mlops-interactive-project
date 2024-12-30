from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'C:/Users/nikam/OneDrive/Desktop/mlops-interactive-project/app/iris_model.pkl')
print("Model trained and saved as 'iris_model.pkl'")
