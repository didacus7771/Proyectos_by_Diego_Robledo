from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess

# ...

def ejecutar_script():
    ruta_script = "/home/didacus7771/proyectos/etl_yt"
    subprocess.run(["python", ruta_script])

default_args = {
    'owner': 'tu_nombre',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'mi_dag',
    default_args=default_args,
    description='Un pipeline simple en Apache Airflow',
    schedule_interval=timedelta(days=1),
)

tarea_ejecutar_script = PythonOperator(
    task_id='ejecutar_script',
    python_callable=ejecutar_script,
    dag=dag,
)

# Otras tareas...

tarea_ejecutar_script  # Asocia la tarea al DAG