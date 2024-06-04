# Utilizar una imagen base oficial de Python
FROM python:3.8-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo de requisitos al contenedor
COPY requirements.txt .

# Instalar las dependencias
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copiar el resto de los archivos del proyecto al contenedor
COPY . .

# Exponer el puerto que usará la aplicación Flask
EXPOSE 80

# Comando para ejecutar la aplicación Flask
CMD ["python", "app.py"]
