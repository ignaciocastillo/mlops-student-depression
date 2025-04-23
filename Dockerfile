# Usa una imagen base oficial con Python
FROM python:3.11-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos necesarios
COPY . .

# Instala las dependencias
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expone el puerto por donde corre Streamlit
EXPOSE 8501

# Comando para ejecutar la app
CMD ["streamlit", "run", "app/streamlit_app.py"]
