import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("parkinsons.csv")
df.columns = df.columns.str.strip()

features = ['PPE', 'MDVP:Jitter(Abs)']
X = df[features]
y = df['status']

x_train, x_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = KNeighborsClassifier(n_neighbors=10)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

joblib.dump(model, "my_model.joblib")





