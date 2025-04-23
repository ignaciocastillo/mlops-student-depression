import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

print("🔄 Loading dataset...")
df = pd.read_csv("data/student_depression.csv")

# ❌ Eliminar columna 'Job Satisfaction' si existe
if 'Job Satisfaction' in df.columns:
    df = df.drop(columns=['Job Satisfaction'])

# ✅ Convertir Sleep Duration
def convert_sleep_duration(duration):
    if pd.isnull(duration):
        return None
    duration = str(duration).replace("'", "").replace("hours", "").strip()
    if '-' in duration:
        try:
            start, end = duration.split('-')
            return (int(start) + int(end)) / 2
        except:
            return None
    if 'Less than' in duration:
        return 4
    if 'More than' in duration:
        return 9
    try:
        return float(duration)
    except:
        return None

df['Sleep Duration'] = df['Sleep Duration'].apply(convert_sleep_duration)

# ❌ Eliminar columnas irrelevantes
drop_cols = ['id', 'City', 'Profession', 'Degree']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# ✅ Codificación de variables categóricas
label_encoder = LabelEncoder()
categorical_cols = ['Gender', 'Dietary Habits', 'Have you ever had suicidal thoughts ?', 'Financial Stress', 'Family History of Mental Illness']
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# ✅ Separar features y target
X = df.drop(columns=['Depression'])
y = df['Depression']

# ✅ División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Entrenamiento
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ✅ Evaluación
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Model retrained. Accuracy: {accuracy * 100:.2f}%")

# ✅ Guardar modelo
joblib.dump(model, "model/modelo_depresion.pkl")
print("💾 Model saved at model/modelo_depresion.pkl")

# Mostrar las columnas utilizadas
print("🧪 Columnas usadas para entrenar el modelo:")
print(X.columns.tolist())
