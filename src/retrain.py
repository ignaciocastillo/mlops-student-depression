import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

print("Loading dataset...")
df = pd.read_csv("data/student_depression.csv")

# Procesar columna 'Sleep Duration'
def convert_sleep_duration(duration):
    if pd.isnull(duration):
        return None
    duration = str(duration).replace("'", "").replace("hours", "").strip()
    if '-' in duration:
        start, end = duration.split('-')
        return (int(start) + int(end)) / 2
    if "Less than" in duration:
        return 4
    if "More than" in duration:
        return 9
    try:
        return float(duration)
    except:
        return None

df['Sleep Duration'] = df['Sleep Duration'].apply(convert_sleep_duration)

# Eliminar columnas irrelevantes si existen
columns_to_drop = ['Job Satisfaction', 'id', 'City', 'Degree', 'Profession']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

# Codificar variables categóricas
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Dietary Habits'] = label_encoder.fit_transform(df['Dietary Habits'])
df['Have you ever had suicidal thoughts ?'] = label_encoder.fit_transform(df['Have you ever had suicidal thoughts ?'])
df['Financial Stress'] = label_encoder.fit_transform(df['Financial Stress'])
df['Family History of Mental Illness'] = label_encoder.fit_transform(df['Family History of Mental Illness'])

# Eliminar filas nulas
df = df.dropna()

# Separar variables
X = df.drop(columns=["Depression"])
y = df["Depression"]

# Entrenar modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model retrained. Accuracy: {accuracy * 100:.2f}%")

# Guardar modelo como diccionario
os.makedirs("model", exist_ok=True)
joblib.dump({"model": model, "accuracy": accuracy}, "model/modelo_depresion.pkl")
print("Model saved.")
