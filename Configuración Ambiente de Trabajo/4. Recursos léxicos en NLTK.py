#!/usr/bin/env python
# coding: utf-8

# In[4]:


import nltk 
nltk.download('book')
from nltk.book import *
from nltk.corpus import stopwords


# # Recursos léxicos (lexicons)
# 
# *   Son colecciones de palabras o frases que tienen asociadas etiquetas o meta-informacion de algún tipo (POS tags, significados gramaticales, etc ...)
# 
# **comentario:** POS (Part of Speech), también llamado etiquetado gramatical o etiquetado de palabras por categorias, consiste en etiquetar la categoria gramatical a la que pertence cada palabra en un volumen de texto, siendo las categorias: 
# 
# 1.   Sustantivos
# 2.   Adjetivos
# 3.   Articulos
# 4.   Pronombres
# 5.   Verbos
# 6.   Adverbios
# 7.   Interjecciones
# 8.   Preposiciones
# 9.   Conjunciones

# **Vocabularios:** palabras únicas en un corpus. Este vocabulario en particular pertenece a la categoría **recursos léxicos no enriquecidos**, ya que solo es la colección de palabras.

# In[5]:


vocab = sorted(set(text1))


# **Distribuciones:** frecuencia de aparición. Con la función `FreqDist` se obtiene una distribución de palabras con objeto tipo diccionario donde las llaves son palabras y los metadatos de esas palabras es la frecuencia de esas palabras. Por lo anteriors se peude decir que es un **recurso léxico (lexicón) enrriquecido**, ya que tiene información sobre las palabras.

# In[6]:


word_freq = FreqDist(text1)


# **Stopwords:** Palabras muy usadas en el lenguaje que usualmente son filtradas en un pipeline de NLP (useless words). Son palabras que se deben **filtrar** para poder hacer una analisis del lenguaje antural significativo.

# In[7]:


print(stopwords.words('spanish'))


# In[8]:


print(stopwords.words('english'))


# ## Fracción de Stopwords en un corpus
# Aplicación sencilla de los **Stopwords** para hacer una limpieza de texto.

# In[9]:


def stopwords_percentage(text):
    '''
    aqui usamos un recurso léxico (stopwords) para filtrar un corpus
    '''
    stopwd = stopwords.words('english')
    content = [w  for w in text if w.lower() not in stopwd]
    return len(content)/len(text)


# Calculamos que porcentaje del libro son **Stopwords** y así podemos definir cual es el volumen de información que debemos procesar en terminos de palabras relevantes.

# In[10]:


stopwords_percentage(text1)


# ## Lexicons enriquecidos (listas comparativas de palabras)
# 
# *   Construyendo diccionarios para traduccion de palabras en diferentes idiomas. 
# 
# 
# Another example of a tabular lexicon is the comparative wordlist. `NLTK` includes so-called **Swadesh** wordlists, lists of about 200 common words in several languages. The languages are identified using an ISO 639 two-letter code.
# 
# 
# * [Accessing Text Corpora and Lexical Resources](https://www.nltk.org/book/ch02.html)
# * [Swadesh](https://www.nltk.org/_modules/nltk/corpus/reader/panlex_swadesh.html)

# In[11]:


from nltk.corpus import swadesh
#idiomas disponibles
print(swadesh.fileids())


# In[12]:


print(swadesh.words('en'))


# Hacemos una traducción del **frances** al **español**

# In[13]:


fr2es = swadesh.entries(['fr', 'es'])
print(fr2es)


# In[14]:


translate = dict(fr2es)
translate['chien']


# In[15]:


translate['jeter']


# # WordNet

# ## Referencias 
# 
# * [WordNet Lecture](https://sp1718.github.io/wordnet_lecture.pdf)
# * [What is WordNet?](https://wordnet.princeton.edu)
# * [WordNet Interface NLTK](http://www.nltk.org/howto/wordnet.html)
# * [LAS-WordNet](https://www.datos.gov.co/Ciencia-Tecnolog-a-e-Innovaci-n/LAS-WordNet-una-WordNet-para-el-espa-ol-obtenida-c/8z8d-85m7)
# 
# Un **synset** es un conjunto de palabras que son sinónimas o que se pueden generalizar con un concepto. Un **synset** se relaciona con otro **synset dependiendo** de la generalidad del concepto.
# 
# La idea del **Wordnet** es tener una estructura tipo grafo.
# 
# ### Conceptos claves:
# 
# * **Hiperonimo:** Es un synset mas generalizado que puede abarcar varias palabras. El ejemplo de la clase es que Artefacto es un hiperónimo de vehículo motorizado.
# * **Hiponimo:** Es un synset que no es general sino más específico.
# 
# Importamos **Wordnet**
# 

# In[16]:


nltk.download('omw')
from nltk.corpus import wordnet as wn


# **synset:** grupo de sinómimos de una palabra.

# In[17]:


ss = wn.synsets('carro', lang='spa')
ss


# Explorando los synsets

# In[18]:


for syn in ss:
  print(syn.name(), ': ', syn.definition())
  for name in syn.lemma_names():
    print(' * ', name)


# ### visualization references
# 
# [Visualizing WordNet relationships as graphs](http://www.randomhacks.net/2009/12/29/visualizing-wordnet-relationships-as-graphs/)
# 
# [Visualizing CIFAR-10 Categories with WordNet and NetworkX](http://dlacombejr.github.io/programming/2015/09/28/visualizing-cifar-10-categories-with-wordnet-and-networkx.html)
# 
# 
# **closure_graph**, construye las relaciones del grafo a paritir de la estrutura del **synset** que se va a solicitar.
# 
# **draw_text_graph**, dibuja el grafo.
# 

# In[25]:


import networkx as nx
import matplotlib.pyplot as plt

def closure_graph(synset, fn):
    seen = set()
    graph = nx.DiGraph()# Objeto vacío, para agregar nodos del grafo
    labels = {}# Objeto vacío para agregar labels

    #Función recursiva que va agregando cada uno de los nodos y labels
    def recurse(s):
        if not s in seen:
            seen.add(s)
            labels[s.name] = s.name().split('.')[0]
            graph.add_node(s.name)
            for s1 in fn(s):
                graph.add_node(s1.name)
                graph.add_edge(s.name, s1.name)
                recurse(s1)

    recurse(synset)
    return graph, labels

def draw_text_graph(G, labels):
    plt.figure(figsize=(18,12))
    pos = nx.planar_layout(G, scale=18)
    nx.draw_networkx_nodes(G, pos, node_color="red", linewidths=0, node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=20, labels=labels)
    nx.draw_networkx_edges(G, pos)
    plt.xticks([])
    plt.yticks([])


# ## **Hyponyms:** 
# Conceptos que son más especificos que la palabra raiz de la cual derivan.

# In[23]:


ss[0].hyponyms()


# In[24]:


print(ss[0].name())
G, labels = closure_graph(ss[0], fn = lambda s: s.hyponyms())
draw_text_graph(G, labels)


# ## **Hypernyms**: 
# conceptos que son mas generales !

# In[27]:


ss[0].hypernyms()


# In[28]:


print(ss[0].name())
G, labels = closure_graph(ss[0], fn = lambda s: s.hypernyms())
draw_text_graph(G, labels)


# ## Similitud Semántica
# 
# La **similitud semántica** en el área de `procesamiento de lenguajes naturales`, es la medida de la interrelación existente entre dos palabras cualesquiera en un texto. Este concepto se fundamenta en la idea que se tiene en `lingüística` sobre la `coexistencia de palabras` y del `discurso coherente`. Dos palabras o términos por el hecho de tener su existencia en un mismo documento poseen un contexto similar. Se entiende que estas dos palabras están relacionadas, y por lo tanto se puede deducir su distancia semántica. 
# 
# **show_syns**, crea la lista de `synsets` y muestra el nombre sel `synset` seguido de la definión.

# In[29]:


def show_syns(word):
  ss = wn.synsets(word, lang='spa')
  for syn in ss:
    print(syn.name(), ': ', syn.definition())
    for name in syn.lemma_names():
      print(' * ', name)
  return ss


# In[30]:


ss = show_syns('perro')


# In[31]:


ss2 = show_syns('gato')


# In[32]:


ss3 = show_syns('animal')


# Se escogen las palabras con las que se va a trabajar deacuerdo a la definción que se quiera utilizar pata calcular la **similitud semántica**, ya que se debe especificar en que contexto se va a realizar el calculo.

# In[39]:


perro = ss[0]
gato = ss2[0]
animal = ss3[0]


# Teniendo las palabras anteriores yu visualizando sus `synsets` vamos a calcular similitud entre '**animal**' y '**perro**'
# 
# En la siguiente intrucción (función **path_similarity**) se esta midiendo el número de vertices que separan los dos `synsets`, calculando un número que refleja una medida de similitud.

# In[41]:


animal.path_similarity(perro)


# Teniendo las palabras anteriores yu visualizando sus `synsets` vamos a calcular similitud entre '**animal**' y '**gato**'

# In[36]:


animal.path_similarity(gato)


# In[37]:


perro.path_similarity(gato)


# Si calculamos similitud entre '**perro**' y '**perro**', su medida es **1** y si son palbras distintas su medida es diferente de **1**. Lo cual quiere decir que la palabra **animal** esta mas cercana a la palabra **perro** que a la palabra **gato** de forma semántica.

# In[42]:


perro.path_similarity(perro)


# **NetworkX** provides basic functionality for visualizing graphs, but its main goal is to enable graph analysis rather than perform graph visualization. In the future, graph visualization functionality may be removed from NetworkX or only available as an add-on package.
# 
# [Drawing](https://networkx.org/documentation/stable//reference/drawing.html)

# In[ ]:


def traverse(graph, start, node):
    graph.depth[node.name] = node.shortest_path_distance(start)
    for child in node.hyponyms():
        graph.add_edge(node.name, child.name)
        traverse(graph, start, child)

def hyponym_graph(start):
    G = nx.Graph()
    G.depth = {}
    traverse(G, start, start)
    return G

def graph_draw(graph):
    nx.draw(graph, 
            node_size = [16*graph.degree(n) for n in graph], 
            node_color = [graph.depth[n] for n in graph], 
            with_labels = False)
    plt.show()


# In[66]:


dog = wn.synset('dog.n.01')
graph = hyponym_graph(dog)
graph_draw(graph)


# In[67]:


dog = wn.synset('dog.n.01')
graph = hyponym_graph(dog)
graph_draw(graph)


# In[ ]:




