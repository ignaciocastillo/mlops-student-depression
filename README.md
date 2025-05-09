# Student Depression Predictor – MLOps Final Project

Este proyecto aplica prácticas de MLOps para entrenar, versionar, desplegar y mantener un modelo de aprendizaje automático que predice el riesgo de depresión en estudiantes, utilizando un conjunto de datos tabulares.

## 🧠 Objetivo

Crear una aplicación completa de machine learning con:
- Reentrenamiento automatizado del modelo
- CI/CD mediante GitHub Actions
- Interfaz de usuario con Streamlit
- Versionamiento de datos/modelos con DVC
- Validación de código con DeepSource
- Publicación funcional en la nube

---

## 📁 Estructura del Proyecto

├── app/
│ └── streamlit_app.py # Interfaz de usuario (Streamlit)
├── data/
│ └── student_depression.csv # Dataset original
├── model/
│ └── modelo_depresion.pkl # Modelo serializado
├── notebooks/
│ └── testing.ipynb # Notebook de pruebas
├── src/
│ └── retrain.py # Script de reentrenamiento
├── .github/workflows/
│ └── retrain.yaml # Workflow CI/CD con GitHub Actions
├── .deepsource.toml # Configuración de análisis estático
├── requirements.txt # Dependencias del proyecto
├── README.md # Este archivo
└── .dvc/ # Metadata de DVC

yaml
Copy
Edit

---

## 🚀 Cómo Ejecutar Localmente

### 1. Clonar el repositorio

```bash
git clone https://github.com/ignaciocastillo/mlops-student-depression.git
cd mlops-student-depression
2. Crear y activar entorno virtual
bash
Copy
Edit
python -m venv .venv
.venv\Scripts\activate  # En Windows
# source .venv/bin/activate  # En Linux/Mac
3. Instalar dependencias
bash
Copy
Edit
pip install -r requirements.txt
4. Descargar el modelo con DVC
bash
Copy
Edit
dvc pull model/modelo_depresion.pkl.dvc
Asegúrese de tener configurado DVC con el remoto S3 habilitado (contactar al dueño del repositorio si no tiene acceso).

5. Ejecutar la app de Streamlit
bash
Copy
Edit
streamlit run app/streamlit_app.py
🔄 Reentrenamiento del Modelo
Para reentrenar el modelo con el archivo data/student_depression.csv:

bash
Copy
Edit
python src/retrain.py
Esto actualiza el archivo modelo_depresion.pkl y puede luego subirlo a S3 y versionarlo con:

bash
Copy
Edit
dvc add model/modelo_depresion.pkl
git add model/modelo_depresion.pkl.dvc
git commit -m "Update model"
dvc push
🧪 ¿Qué hace testing.ipynb?
Este notebook se utiliza para probar la carga del modelo serializado y realizar predicciones con nuevas muestras. Sirve como demostración técnica para validar que el modelo funciona fuera del flujo automático, usando entradas simuladas para observar su desempeño.

🔍 Validación de Código
Se utiliza DeepSource para:

Detectar errores de estilo y bugs

Eliminar código muerto o sin usar

Verificar convenciones de PEP8 y seguridad

⚙️ CI/CD con GitHub Actions
El archivo .github/workflows/retrain.yaml permite:

Reentrenar automáticamente el modelo al hacer push en main

Subir el modelo a DVC

Validar con flake8 y DeepSource

✅ Checklist de Entregables
 Modelo serializado

 Reentrenamiento automático (retrain.py)

 Streamlit app funcional

 CI/CD con GitHub Actions

 Validación estática con DeepSource

 Versionamiento de datos y modelo con DVC

 README completo y explicativo

 Pruebas en testing.ipynb

 Publicación funcional en Streamlit Cloud

🌐 Enlace a la aplicación
🔗 Ver aplicación en Streamlit

🧠 Autores
Ignacio Castillo Vega