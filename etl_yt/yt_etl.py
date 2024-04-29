# Para la estructura del DataFrame

import pandas as pd

# Para interactuar con la API de Youtube

from googleapiclient.discovery import build


# Bibliotecas de autenticacion para Google Api

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# Para interactuar con Amazon S3

import s3fs


# Configurar la clave de API de Youtube

api_key = 'AIzaSyBcrB9i03XVofwqyYoN5H0MB58qmNuQ2wA'

# Funcion para obtener todos los comentarios de un video

from googleapiclient.discovery import build
import pandas as pd

def get_video_comments(api_key, video_id):
    youtube = build('youtube', 'v3', developerKey=api_key)

    users = []
    comments = []
    next_page_token = None

    while True:
        comments_response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            textFormat='plainText',
            maxResults=100,
            pageToken=next_page_token
        ).execute()

        # Recorrer los comentarios y agregar información del usuario y comentario a las listas
        for item in comments_response['items']:
            user_info = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
            comment_text = item['snippet']['topLevelComment']['snippet']['textDisplay']
            users.append(user_info)
            comments.append(comment_text)

        # Verificar si hay más páginas de comentarios
        next_page_token = comments_response.get('nextPageToken')
        if not next_page_token:
            break

    return users, comments  # Devolver dos listas separadas para usuarios y comentarios

if __name__ == "__main__":
    # Ejemplo de uso
    users, comments = get_video_comments('AIzaSyBcrB9i03XVofwqyYoN5H0MB58qmNuQ2wA', 'wxjxNE1Ddng')

    # Crear el DataFrame
    df = pd.DataFrame({'Usuario': users, 'Comentario': comments})
    df.to_csv('comentarios_video_dross.csv')


# Imprimir los nombres de usuarios y comentarios
# for i, data in enumerate(video_data, start=1):
    #user = data['user']
    #comment = data['comment']
    #print(f"Usuario {i}: {user}\nComentario: {comment}\n")
    




