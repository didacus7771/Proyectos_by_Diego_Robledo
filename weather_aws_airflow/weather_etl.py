import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import boto3
from botocore.exceptions import NoCredentialsError

def obtener_datos_temperatura(api_key, ciudad, fecha_inicio, dias):
    # Formatear la fecha de inicio en el formato necesario para la API
    fecha_inicio_str = fecha_inicio.strftime('%Y-%m-%d')

    # Obtener la fecha final sumando los días especificados
    fecha_fin = fecha_inicio + timedelta(days=dias)
    fecha_fin_str = fecha_fin.strftime('%Y-%m-%d')

    # Construir la URL de la API para obtener el pronóstico de los próximos días
    api_url = f'http://api.openweathermap.org/data/2.5/forecast?q={ciudad}&appid={api_key}&start={fecha_inicio_str}&end={fecha_fin_str}&units=metric'

    # Realizar la solicitud a la API
    response = requests.get(api_url)
    data = json.loads(response.text)

    # Crear un DataFrame con los datos relevantes
    df = pd.DataFrame({
        'Fecha': [registro['dt_txt'] for registro in data['list']],
        'Temperatura (Celsius)': [registro['main']['temp'] for registro in data['list']],
    })

    return df

def guardar_datos_en_csv(df, nombre_archivo):
    df.to_csv(nombre_archivo, index=False)

def cargar_en_s3(archivo_local, nombre_bucket, nombre_objeto):
    s3 = boto3.client('s3')

if __name__ == "__main__":
    clave_api = 'b61191a65bdec93a25b88dcffb181182'
    ciudad_buenos_aires = 'Buenos Aires, AR'
    fecha_inicio = datetime(2024, 2, 27)  # Establecer la fecha de inicio
    nombre_archivo_local = 'temperatura_Buenos_Aires.csv'
    datos_temperatura = obtener_datos_temperatura(clave_api, ciudad_buenos_aires, fecha_inicio, dias=7)
    guardar_datos_en_csv(datos_temperatura, 'temperatura_Buenos_Aires.csv')

 # Configura solo el nombre del bucket y del objeto S3
    nombre_bucket_s3 = 'project-airflow-aws-weather'
    nombre_objeto_s3 = 'temperatura_Buenos_Aires.csv'

    cargar_en_s3(nombre_archivo_local, nombre_bucket_s3, nombre_objeto_s3)
