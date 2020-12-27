#!/usr/bin/env python
# coding: utf-8

# # Configuración Inicial

# In[1]:


import nltk 
nltk.download('book')
from nltk.book import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px


# # Bi-gramas
# * Parejas de palabras que ocurren consecutivas.

# In[2]:


md_bigrams = list(bigrams(text1))
fdist = FreqDist(md_bigrams)
md_bigrams[:10]


# Aqui podemos obtener los bi-gramas más frecuentes en este texto:

# In[3]:


fdist.most_common(10)


# In[4]:


fdist.plot(20)


# ## Filtrado de bi-gramas
# * Sin embargo, observamos que los bi-gramas más comunes no representan realmente frases o estructuras léxicas de interes.
# * Tal vez, aplicar algun tipo de filtro nos permita ver estructuras más relevantes.

# In[5]:


threshold = 2
filtered_bigrams = [bigram for bigram in md_bigrams if len(bigram[0])>threshold and len(bigram[1])>threshold]
filtered_dist = FreqDist(filtered_bigrams)
filtered_dist.plot(20)


# # Tri-gramas

# In[6]:


from nltk.util import ngrams


# In[7]:


md_trigrams = list(ngrams(text1,3))
fdist = FreqDist(md_trigrams)
md_trigrams[:10]


# In[8]:


fdist.most_common(10)


# In[9]:


fdist.plot(20)


# # Collocations (Colocaciones)
# 
# * Son secuencias de palabras que suelen ocurrir en textos o conversaciones con una **frecuencia inusualmente alta** [NLTK doc](http://www.nltk.org/book/ch01.html)
# * Las colocaciones de una palabra son declaraciones formales de donde suele ubicarse tipicamente esa palabra [Manning & Schütze, 1990, Foundations of Statistical Natural Language Processing, Capítulo 6](https://nlp.stanford.edu/fsnlp/)

# In[10]:


md_bigrams = list(bigrams(text1))


# In[11]:


threshold = 2
#distribution of bi-grams
filtered_bigrams = [bigram for bigram in md_bigrams if len(bigram[0])>threshold and len(bigram[1])>threshold]
filtered_bigram_dist = FreqDist(filtered_bigrams)


# In[12]:


#distribution of words
filtered_words = [word for word in text1 if len(word)>threshold]
filtered_word_dist = FreqDist(filtered_words)


# Tener presente que `pd.DataFrame()` es una hoja de Excel en versión Python, lo cual implica que funciona igual que una hoja de Excel.
# 
# * **`df['bi_gram']`** muestra los bigramas que estan contenidos en  `filtered_bigrams`
# * **`df['word_0']`** muestra la primera palabra de cada bigrama (Tupla) en `filtered_bigrams`
# * **`df['word_1']`** muestra la segunda palabra de cada bigrama (Tupla) en `filtered_bigrams`
# * **`df['bi_gram_freq']`** muestra la frecuencia de aparición de cada bigrama en `filtered_bigrams` con `filtered_bigram_dist`
# * **`df['word_0_freq']`** muestra la frecuencia de aparición de cada palabra de la primera posición del bigrama en `filtered_words` con `filtered_word_dist`
# * **`df['word_1_freq']`** muestra la frecuencia de aparición de cada palabra de la segunda posición del bigrama en `filtered_words` con `filtered_word_dist`

# In[13]:


df = pd.DataFrame()
df['bi_gram'] = list(set(filtered_bigrams))
df['word_0'] = df['bi_gram'].apply(lambda x: x[0])
df['word_1'] = df['bi_gram'].apply(lambda x: x[1])
df['bi_gram_freq'] = df['bi_gram'].apply(lambda x: filtered_bigram_dist[x])
df['word_0_freq'] = df['word_0'].apply(lambda x: filtered_word_dist[x])
df['word_1_freq'] = df['word_1'].apply(lambda x: filtered_word_dist[x])
df


# # Pointwise Mutual Information (PMI)
# Una métrica basada en _teoria de la información_ para encontrar **Collocations**.
# 
# $$
# PMI = \log\left(\frac{P(w_1, w_2)}{P(w_1)P(w_2)}\right)
# $$

# El Punto de **Información Mutua** (PIM) o **Información Mutua Puntual**, (IMP) (en inglés, **Pointwise mutual information** (PMI)), es una medida de asociación utilizada en la teoría y la estadística de la información. En contraste con la información mutua (Mutual Information, MI), que se basa en PIM, esta se refiere a los eventos individuales, mientras que MI se refiere a la media de todos los eventos posibles.
# 
# En **lingüística computacional**, PMI ha sido usado para encontrar colocaciones y asociaciones entre palabras. Por ejemplo, los conteos de occurrencias y co-ocurrencias de las palabras en un corpus puede ser usado para aproximar las probabilidades **p(x)** y **p(x, y)** respectivamente.
# 
# El `PMI` siempre es un número menor ó igual a cero, ya que la división suele dar un número inferior a `1`, por lo cual el logaritmo es negativo.
# 
# El `PMI` no es suficiente para encontrar bigramas representativos, ya que los bigramas con mayor `PMI` tiene frecuencias muy bajas, para poder encontrar bigramas interesantes, hay que utilizar la frecuencia del brigrama `bi_gram_freq` como valor del eje **y**, para construir un gráfica, lo cual implica aplicarle **log** para que quede en la misma unidad de medida que **PMI**

# In[14]:


df['PMI'] = df[['bi_gram_freq', 'word_0_freq', 'word_1_freq']].apply(lambda x:np.log2(x.values[0]/(x.values[1]*x.values[2])), axis = 1)
df['log(bi_gram_freq)'] = df['bi_gram_freq'].apply(lambda x: np.log2(x))
df


# Cambiamos la visualización del Dataframe organizando por la cololumna `PMI` mostrando los valores más grandes en primer lugar.

# In[15]:


df.sort_values(by = 'PMI', ascending=False)


# ### Creamos un gráfico de dispersión
# * **x** los valores en x
# * **y** los valores en y
# * **color** el color de los puntos, el cual se calcula con base a la proporción de las variables (Las sumamos para este de caso)
# * **size** los valores de esta columna se utilizan para asignar el tamaño al punto del grafico, para este caso se calcula la proporción de la suma de de `` y `` donde se evidencia que a menor valor en la suma el punto es más grande y por tal motivo más representativo.
# * **hover_name** mustra que se ve cuando se pasa el cursos sobre un punto
# * **width** el ancho del gráfico
# * **height** el alto del gráfico
# * **labels** los valores del las etiquetas
# 
# [plotly.express.scatter](https://plotly.com/python-api-reference/generated/plotly.express.scatter)

# In[16]:


fig = px.scatter(x = df['PMI'].values, y = df['log(bi_gram_freq)'].values, color = df['PMI']+df['log(bi_gram_freq)'], 
                 size = (df['PMI']+df['log(bi_gram_freq)']).apply(lambda x: 1/(1+abs(x))).values, 
                 hover_name = df['bi_gram'].values, width = 1000, height = 800, labels = {'x': 'PMI', 'y': 'Log(Bigram Frequency)'})
fig.show()


# # Medidas pre-construidas en NLTK

# Ver la documentación de [Ngram Association Measures](https://www.nltk.org/_modules/nltk/metrics/association.html)
# 
# Provides scoring functions for a number of association measures through a generic, abstract implementation in `NgramAssocMeasures`, and n-specific `BigramAssocMeasures` and `TrigramAssocMeasures`.
# 
# La métrica `PMI`, ya esta implementada dentro de `NLTK` dentro del objeto **nltk.collocations.BigramAssocMeasures()**
# 
# A continuación se importa el método **BigramAssocMeasures()** de `NLTK` quew permite usar todas las metricas y **BigramCollocationFinder.from_words(x)** es un metodo que nos permite implementar una clase para las colocaciones.

# In[17]:


from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()


# Documentación de [BigramCollocationFinder](https://tedboy.github.io/nlps/generated/generated/nltk.BigramCollocationFinder.html)

# In[18]:


finder = BigramCollocationFinder.from_words(text1)


# **apply_freq_filter(F)** es un filtro de frecuencia, donde se especifica que deje los bigramas que tengan una frecuencia superior a `F`
# 
# **nbest(score_fn, n)** retorna los mejores candidatos a colocaciones con base a la función dada y requiere dos argumentos **(función de puntuación, número de colocaciones)**
# 
# la **función de puntuación** puede ser cualquier métrica obtenida de **nltk.collocations.BigramAssocMeasures()** ó **nltk.collocations.TrigramAssocMeasures()**
# 
# * bigram_measures.pmi
# * bigram_measures.likelihood
# * trigram_measures.raw_freq
# * etc...

# In[19]:


finder.apply_freq_filter(20)
finder.nbest(bigram_measures.pmi, 10)


# # Textos en Español 

# Descargamos corpus en español [Spanish corpus](https://mailman.uib.no/public/corpora/2007-October/005448.html)

# In[20]:


nltk.download('cess_esp')
corpus = nltk.corpus.cess_esp.sents() 
flatten_corpus = [w for l in corpus for w in l]


# In[21]:


print(corpus[:2])


# In[22]:


print(flatten_corpus[:50])


# Documentación de [BigramCollocationFinder](https://tedboy.github.io/nlps/generated/generated/nltk.BigramCollocationFinder.html)

# In[23]:


finder = BigramCollocationFinder.from_documents(corpus)
finder.apply_freq_filter(10)
finder.nbest(bigram_measures.pmi, 10)


# # Referencias para seguir aprendiendo
# 
# 
# *   [Mas sobre Colocaciones con NLTK](http://www.nltk.org/howto/collocations.html)
# 
# 

# In[ ]:




