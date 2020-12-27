#!/usr/bin/env python
# coding: utf-8

# # Configuración inicial

# In[42]:


import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk import word_tokenize
from urllib import request


# # Procesar texto plano desde Web

# In[43]:


url = "http://www.gutenberg.org/files/2554/2554-0.txt"
response = request.urlopen(url)
raw = response.read().decode('utf8')


# In[44]:


raw


# In[45]:


with open('book.txt', 'w')as f:
    f.write(raw)


# In[46]:


len(raw)


# In[47]:


tokens = word_tokenize(raw)
print(tokens[:20])


# La instrucción **collocations()** puede generar el siguiente error: <font color='red'>ValueError:</font> `too many values to unpack (expected 2)`
# 
# Esto se debe a un bug (Reporte del bug [aquí](https://github.com/nltk/nltk_book/issues/224)) en la librería y para solucionarlo hay que cambiar la instrucción. Es posible que el error se presene ebido a la versión utilizada de la librería.
# 
# > text.collocations()
# 
# por
# 
# > print('; '.join(text.collocation_list()))

# In[48]:


text = nltk.Text(tokens)
#text.collocations()
print('; '.join(text.collocation_list()))


# Lo anterior corresponde a las colocaciones en particular del libro **Crime and Punishment**, cuya frecuencia de aparición es inusualmente alta!  Generalmente, esas colocaciones (bi-gramas, tri-gramas, n-gramas) y ocasionalmente corresponden a nombres propios o expresiones particulares del idioma donde se estan utilizando.

# # Procesar HTML 

# In[49]:


import requests
from bs4 import BeautifulSoup
import re
from nltk.tokenize import RegexpTokenizer


# In[50]:


url = 'https://www.gutenberg.org/files/2701/2701-h/2701-h.htm'
r = requests.get(url)


# In[51]:


html = r.text
html 


# `HTML` el `HTML` colocando la jerarquía de tabs HTML que se estan utilzando.

# In[52]:


soup = BeautifulSoup(html, 'html.parser')
soup


# In[53]:


text = soup.get_text()


# Usamos expresiones regulares para quetar los tags `HTML` y obtener el texto relavante. sin embargo todavía aparecera texto que **NO** es relvante como `H1, H2, H3, H4, etc.`

# In[54]:


tokens = re.findall('\w+', text)
tokens[:50]


# Podemos utilizar otro metodo para obtener los **tokens** y además les aplicamos `lower case`
# 
# **RegexpTokenizer** A tokenizer that splits a string using a regular expression, which matches either the tokens or the separators between tokens. [RegexpTokenizer](https://www.kite.com/python/docs/nltk.RegexpTokenizer)

# In[55]:


tokenizer = RegexpTokenizer('\w+')
tokens = tokenizer.tokenize(text)
tokens = [token.lower() for token in tokens]
tokens[:50]


# In[ ]:


text = nltk.Text(tokens)
#text.collocations()
print('; '.join(text.collocation_list()))


# El script llamado [html2text](https://github.com/Alir3z4/html2text) sirve para convertir páginas HTML en texto plano (eliminando todos los tags de html) y así obtener tokens que corresponden únicamente al contenido del libro. A continuación les muestro cómo instalarlo en Google Colab y obtener los tokens. Funciona de maravilla!

# In[ ]:


import html2text


# In[ ]:


to_text = html2text.html2text(html)
print(to_text)


# In[ ]:


tokens = re.findall('\w+', to_text)
tokens[:50]


# In[ ]:





# In[56]:





# El script llamado [html2text](https://github.com/Alir3z4/html2text) sirve para convertir páginas HTML en texto plano (eliminando todos los tags de html) y así obtener tokens que corresponden únicamente al contenido del libro. A continuación les muestro cómo instalarlo en Google Colab y obtener los tokens. Funciona de maravilla!

# In[57]:


import html2text


# In[58]:


to_text = html2text.html2text(html)
print(to_text)


# In[59]:


tokens = re.findall('\w+', to_text)
tokens[:50]


# In[ ]:




