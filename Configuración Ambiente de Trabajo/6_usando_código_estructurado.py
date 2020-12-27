# -*- coding: utf-8 -*-
"""6. Usando código estructurado.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16fpHZl8Ymvw6e4FKPtkcozjb1XFSm-8K

# Configuracion inicial
"""

from google.colab import drive
drive.mount('/content/drive')
filepath = '/content/drive/My Drive/Colab Notebooks/NLP_course_resources/'

"""# Construyendo codigo estructurado (con funciones)

## Ejemplo 1:
"""

#Importamos libería de regular expressions
import re

# la funcion la podemos definir en el notebook y usar directamente
def get_text(file):
  '''Read Text from file'''
  text = open(file).read()
  text = re.sub(r'<.*?>', ' ', text)
  text = re.sub(r'\s+', ' ', text)
  return text

text = get_text(filepath+'book.txt')
text

from pathlib import Path

if not Path(filepath+'JaidenRead.py').is_file():
    print ("File not exist")
    # será mas comodo definir la funcion dentro de una libreria externa
    !touch /content/drive/My\ Drive/Colab\ Notebooks/NLP_course_resources/JaidenRead.py
else:
    print ("File exist")

#Importamos libería del sistema, para agregar ruta repositorio de liberías
import sys
#Limpiamos el path del sistema si previamente agregamos nuestra carpeta
try:
  sys.path.remove(filepath)
except:
  print("El path personalizado no exixte!") 

sys.path

#Incluimos nuestra carpeta como repositorio de libreríase
sys.path.append(filepath)
sys.path

import JaidenRead
JaidenRead.get_text(filepath+'book.txt')

"""# Ejemplo 2: """

import nltk
nltk.download('punkt')
from urllib import request
from bs4 import BeautifulSoup
from nltk import word_tokenize

# Obtenemos la lista de palabras cons sus respectivas frecuencias
# y hacemos la implementación en una sola función
def freq_words(url, n, encoding = 'utf8'):
  req = request.urlopen(url)
  html = req.read().decode(encoding)
  raw = BeautifulSoup(html, 'html.parser')
  text = raw.get_text()
  tokens = word_tokenize(text)
  tokens = [t.lower() for t in tokens]
  fd = nltk.FreqDist(tokens)
  return [t for (t, _) in fd.most_common(n)]

freq_words('https://www.gutenberg.org/files/2701/2701-h/2701-h.htm', 20)

if not Path(filepath+'nlp_JaidenUtils.py').is_file():
  print ("File not exist")
  # será mas comodo definir la funcion dentro de una libreria externa
  !touch /content/drive/My\ Drive/Colab\ Notebooks/NLP_course_resources/nlp_JaidenUtils.py
else:
  print ("File exist")

import nlp_JaidenUtils
nlp_JaidenUtils.freq_words('https://www.gutenberg.org/files/2701/2701-h/2701-h.htm', 20)

