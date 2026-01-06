import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("parkinsons.csv")

# Feature engineering
df['freq_range'] = df['MDVP:Fhi(Hz)'] - df['MDVP:Flo(Hz)']
df['jitter_shimmer_ratio'] = df['MDVP:Jitter(%)'] / (df['MDVP:Shimmer'] + 1e-6)
df['noise_to_harmonic'] = 1 / (df['HNR'] + 1e-6)
df['complexity_score'] = df['RPDE'] + df['DFA']

# Selected features
selected_features = [
    'PPE',
    'HNR',
    'freq_range',
    'jitter_shimmer_ratio',
    'noise_to_harmonic',
    'complexity_score'
]

X = df[selected_features]
y = df['status']

# Train / test split
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42
)

model.fit(x_train, y_train)

# Evaluation
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save model
joblib.dump(model, "parkinsons_rf.joblib")



