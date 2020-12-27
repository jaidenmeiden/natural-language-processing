#!/usr/bin/env python
# coding: utf-8

# # **Como usar NLTK**

# In[1]:


#seleccionar download [d], luego descargar el recurso de nombre "book"
import nltk
nltk.download('cess_esp')


# # **Expresiones Regulares**
# 
# 
# *   Constituyen un lenguaje estandarizado para definir cadenas de búsqueda de texto.
# *   Libreria de operaciones con  expresiones regulares de Python [re](https://docs.python.org/3/library/re.html)
# *   Reglas para escribir expresiones regulares [Wiki](https://es.wikipedia.org/wiki/Expresión_regular)
# 
# 

# In[2]:


# spanish Corpus: https://mailman.uib.no/public/corpora/2007-October/005448.html
import re
corpus = nltk.corpus.cess_esp.sents()
print(corpus)
print(len(corpus))


# Hay que tener presente qu en python la siguiente intrucción:
# 
# > l: lists into corpus
# 
# > w: words into list
# 
# ```
# w for l in corpus for w in l
# ```
# Es exactamente igual a:
# 
# ```
# for l in corpus:
#     for w in l:
# ```
# 
# 

# In[3]:


flatten = [w for l in corpus for w in l]
# Total de tokens, los cuales pueden ser palabras, signos de puntuación, etc...
print(len(flatten))


# In[4]:


print(flatten[:20])


# ## **Meta-caracteres básicos**

# ### **Estructura de la funcion re.search()**
# 
# Determina si el patron de búsqueda p esta contenido en la cadena s
# ```
# re.search(p, s)
# ```
# 
# 

# Busca el meta caracter dentro de cada palabra

# In[5]:


array = [w for w in flatten if re.search('es', w)]
print(array[:5])


# Busca el meta-caracter al final de cada palabra

# In[6]:


array = [w for w in flatten if re.search('es$', w)]
print(array[:5])


# Busca el meta-caracter al principio de cada palabra

# In[7]:


array = [w for w in flatten if re.search('^es', w)]
print(array[:5])


# Busca meta-caracteres siguiendo una secuencia

# In[8]:


array = [w for w in flatten if re.search('^..j..t..$', w)]
print(array[:5])


# ### **Rangos [a-z], [A-Z], [0-9]**

# Busca todas las palabras que comiencen por las legras **g**, **h** ó **i**

# In[9]:


array = [w for w in flatten if re.search('^[ghi]', w)]
print(array[:10])


# Busca todas las palabras que terminen por las legras **a**, **e** ú **o**

# In[10]:


array = [w for w in flatten if re.search('[aeo]$', w)]
print(array[:10])


# Busca todas las palabras cuya primera letra sea **g**, **h** o **i**, la segunda letra sea **m**, **n** ú **o**. la tercera letra sea **j**, **l** o **k** y la ultima sea **d**, **e** o **f**. Explora todas las posibles combinaciones.

# In[11]:


array = [w for w in flatten if re.search('^[ghi][mno][jlk][def]$', w)]
array


# ### **Clausuras (Kleene closures)**

# Secciones de cadena de texto que se pueden repetir
# 
# La clausura con ***** repite 0 o más veces
# 
# La clausura con **+** repite 1 o más veces

# Busca todas las palabras en las que se repita la secuencia **no** y que se repita 0 (cero quiere decir que no aparezca) o más veces

# In[12]:


array = [w for w in flatten if re.search('^(no)*', w)]
array[:10]


# Busca todas las palabras en las que se repita la secuencia **no** y que se repita 1 o más veces

# In[13]:


array = [w for w in flatten if re.search('^(no)+', w)]
array[:10]


# # **Normalización de Texto**
# Como aplicación de las expresiones regulares
# 

# In[14]:


print('Esta es \n una prueba!')


# ### **raw**

# In[15]:


# Al colocar la letra r al pricipio de una cadena de texto, 
# interpreta la cadena de texto como texrto plano
print(r'Esta es \n una prueba!')


# ## **Tokenización:** 
# 
# Es el proceso mediante el cual se sub-divide una cadena de texto en unidades linguísticas minimas (palabras)
# 

# In[16]:


texto = """ Cuando sea el rey del mundo  (imaginaba él en su cabeza) no tendré que  preocuparme por estas bobadas. 
            Era solo un niño de 7 años, pero pensaba que podría ser cualquier cosa que su imaginación le permitiera visualizar en su cabeza ..."""
print(texto)


# **Caso 1:** tokenizacion más simple: por espacios vacios!
# 

# In[17]:


print(re.split(r' ', texto))


# **Caso 2:** tokenización usando expresiones regulares [Espacio vacio], [tabulaciones] y [nuevas lineas]
# 

# In[18]:


print(re.split(r'[ \t\n]+', texto))


# **Caso 3:** tokenización usando expresiones regulares [Espacio vacio], [todos los caracteres que no sean letras, dígitos o guiones bajos], [tabulaciones] y [nuevas lineas]
# 
# **RegEx reference:** \W -> all characters other than letters, digits or underscore
# 

# In[19]:


print(re.split(r'[ \W\t\n]+', texto))


# ## **Tokenizador de NLTK**

# Nuestra antigua regex no funciona en este caso, ya que no tiene en cuenta la abreviación de Estados Unidos y tampos el valos monetario como tokens independientes.
# 

# In[20]:


print(re.split(r'[ \W\t\n]+', texto))


# A continuación se expone un patrón más complejo en el cual se inclyen varias reglas, las cuales peuden ser modificadas ó agregar más según los requermientos.
# 

# In[21]:


pattern = r'''(?x)                 # set flag to allow verbose regexps
              (?:[A-Z]\.)+         # abbreviations, e.g. U.S.A.
              | \w+(?:-\w+)*       # words with optional internal hyphens
              | \$?\d+(?:\.\d+)?%? # currency and percentages, e.g. $12.40, 82%
              | \.\.\.             # ellipsis
              | [][.,;"'?():-_`]   # these are separate tokens; includes ], [
'''
nltk.regexp_tokenize(texto, pattern)


# ## **Lematización:** 
# 
# Proceso para encontrar la raíz linguística de una palabra
# 
# *   Derivación (stemming) : lematización simple

# In[26]:


# Derivación simple
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
SnowballStemmer.languages


# In[27]:


stem = SnowballStemmer('spanish')
stem.stem('trabajando')


# In[28]:


# Lematización
from nltk.stem import WordNetLemmatizer
lemm = WordNetLemmatizer()


# In[29]:


lemm.lemmatize('trabajando')


# In[ ]:




