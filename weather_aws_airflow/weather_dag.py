from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import json
import pandas as pd
from datetime import datetime, timedelta

default_args = {
    'owner': 'didacus',
    'start_date': datetime(2024, 2, 27),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def obtener_datos_temperatura(api_key, ciudad, fecha_inicio, dias):
    fecha_inicio_str = fecha_inicio.strftime('%Y-%m-%d')
    fecha_fin = fecha_inicio + timedelta(days=dias)
    fecha_fin_str = fecha_fin.strftime('%Y-%m-%d')
    api_url = f'http://api.openweathermap.org/data/2.5/forecast?q={ciudad}&appid={api_key}&start={fecha_inicio_str}&end={fecha_fin_str}&units=metric'
    response = requests.get(api_url)
    data = json.loads(response.text)
    df = pd.DataFrame({
        'Fecha': [registro['dt_txt'] for registro in data['list']],
        'Temperatura (Celsius)': [registro['main']['temp'] for registro in data['list']],
    })

    return df

def guardar_datos_en_csv(df, nombre_archivo):
    df.to_csv(nombre_archivo, index=False)

def ejecutar_script(**kwargs):
    clave_api = 'b61191a65bdec93a25b88dcffb181182'
    ciudad_buenos_aires = 'Buenos Aires, AR'
    fecha_inicio = datetime(2024, 2, 27)
    datos_temperatura = obtener_datos_temperatura(clave_api, ciudad_buenos_aires, fecha_inicio, dias=7)
    guardar_datos_en_csv(datos_temperatura, 'temperatura_Buenos_Aires.csv')

dag = DAG(
    'mi_dag',
    default_args=default_args,
    description='Temperatura de Buenos Aires de los proximos días',
    schedule_interval=timedelta(days=1),  # Frecuencia de ejecución
)

ejecutar_script_task = PythonOperator(
    task_id='ejecutar_script',
    python_callable=ejecutar_script,
    provide_context=True,  # Proporciona el contexto para acceder a las variables de la tarea
    dag=dag,
)

ejecutar_script_task