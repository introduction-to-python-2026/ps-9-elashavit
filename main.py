import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("parkinsons.csv")
df.columns = df.columns.str.strip()

# Select features (exactly 2 for the Test Runner)
features = ['PPE', 'MDVP:Jitter(Abs)']
X = df[features]
y = df['status']

# Split into train/test
x_train, x_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = KNeighborsClassifier(n_neighbors=10)
model.fit(x_train, y_train)

# Evaluate
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the trained model (will create my_model.joblib)
joblib.dump(model, "my_model.joblib")
print("my_model.joblib saved successfully")
