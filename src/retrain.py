import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

print("🔄 Loading dataset...")
df = pd.read_csv("data/student_depression.csv")

# Limpieza de columna Sleep Duration
def convert_sleep_duration(duration):
    if isinstance(duration, str):
        duration = duration.replace("'", "").replace("hours", "").strip()
        if '-' in duration:
            start, end = duration.split('-')
            return (int(start) + int(end)) // 2
        if 'Less than' in duration:
            return 4
        if 'More than' in duration:
            return 9
        try:
            return int(float(duration))
        except ValueError:
            return None
    return duration

df["Sleep Duration"] = df["Sleep Duration"].apply(convert_sleep_duration)

# Eliminar filas con valores nulos
df.dropna(inplace=True)

# Codificación de columnas categóricas
label_encoder = LabelEncoder()
df["Gender"] = label_encoder.fit_transform(df["Gender"])
df["Dietary Habits"] = label_encoder.fit_transform(df["Dietary Habits"])
df["Have you ever had suicidal thoughts ?"] = label_encoder.fit_transform(df["Have you ever had suicidal thoughts ?"])
df["Financial Stress"] = label_encoder.fit_transform(df["Financial Stress"])
df["Family History of Mental Illness"] = label_encoder.fit_transform(df["Family History of Mental Illness"])

# Eliminar columnas irrelevantes o no numéricas
columns_to_drop = ["id", "City", "Degree", "Profession"]
for col in columns_to_drop:
    if col in df.columns:
        df.drop(columns=col, inplace=True)


# Features y label
X = df.drop("Depression", axis=1)
y = df["Depression"]

# División y entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluación
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model retrained. Accuracy: {accuracy * 100:.2f}%")

# Guardar modelo
joblib.dump(model, "model/modelo_depresion.pkl")
print("💾 Model saved at model/modelo_depresion.pkl")