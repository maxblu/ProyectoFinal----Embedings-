import streamlit as st
from nltk.corpus import cess_esp
from gensim.models import Word2Vec
import multiprocessing

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy import spatial
import numpy as np

import plotly.express as px
import plotly.graph_objects as go


# val = st.radio("",["Info","Entrenando y visualizando embedings","Usando embeding preentrenados"])
val = st.sidebar.selectbox("",["Info","Entrenando y visualizando embedings","Usando embeding preentrenados"])
# info = st.button("Info")
# trainn = st.button("Entrenando y visualizando embedings")
# pretra = st.button("Usando embeding preentrenados")

if val == "Info":

        st.header("Word Embedings")

        st.subheader("Objetivos:")

        "* Introducir los *word embedings* como forma de representar texto y como difieren estos de las otras alternativas."
        " * Principales algoritmos para aprender estas representaciones."
        " * Mostrar el uso de modelos pre-entrenados de estas representaciones "

        """
        Ya hemos visto otras representaciones de textos en formas vectorial, como por ejemplo *one-hot* o las representaciones usando        *tf-idf*, el problema de estas es que muchas veces son densas, de grandes dimensiones o simplemente carecen del poder para captar la   semántica de las palabras en un contexto dado así como las relaciones entre ellas.

        Para intentar resolver estos problemas surge los *word embedings* como una nueva forma de representar los las palabras en forma de      vectores de números reales. Es una representación aprendida de texto en el que las palabras que tienen el mismo significado tienen una       representación similar.  Cada palabra es mapeada a un vector de valores que son aprendidos mediante una red neuronal profunda.

        Existen diferentes algoritmos para aprender estas representaciones, como son *Word2Vec*, *GloVe* y *BERT*. 

        *** Word2Vec  ***

        Es un método estadístico para aprender los *embedings* desarrollado por Tomas Mikolov. El análisis de los vectores aprendidos y la      exploración de la matemática vectorial en la representción de palabras  permitión o lograr cosas como que substrayendo al vector ¨Rey¨       el de la palabra ¨hombre¨ se obtenía un vector cercano a de ¨reina¨.

        Existen dos modelos para aprender estas representaciones siguiendo este método:
        """
        "* Continuous Bag-of-Words, or CBOW "
        "* Continuous Skip-Gram "

        """
        La arquitectura skip-gram está estructurada como un clasificador que toma como entrada dos palabras del vocabulario: p y c, e intenta   predecir si la palabra c debería aparecer en el contexto de la palabra p. Concretamente, la entrada de dicha arquitectura son las         representaciones one-hot 1 de las palabras p y c. Para cada una de estas entradas hay una capa intermedia cuya dimensión de salida es la        dimensión del embedding que se desea obtener. Luego el producto escalar de los vectores de salida de estas capas sirve de entrada a una        neurona que actúa como clasificador en la capa de salida. Luego de optimizar este modelo, el embedding de una palabra se obtiene a     partir de la matriz que se corresponde con la capa de embedding asociada a la palabra de entrada p. Dicha matriz tiene dimensiones V X E,    siendo V el tamaño del vocabulario y E la dimensión del embedding. En la misma, la fila i-ésima constituye el vector que representa a     la palabra con índice i en el vocabulario. 

        La arquitectura CBOW por su parte está estructurada como un clasificador multiclase que toma como entrada un cojunto de términos de     tamaño fijo C, que representa el posible contexto de una palabra, e intenta predecer cúal es la palabra que p que tiene a C como    contexto. Concretamente, la entrada de dicha arquitectura es una lista de vectores one-hot que represetan a las palabras en C. Cada uno    de los cuales pasa por una misma capa de embedding. Después, la lista de vectores correspondiente a las salidas de esta capa se proyecta   como un solo vector (usualmente para ello se
        utiliza el centroide de los vectores). Y este último alimenta una capa densa con V neuronas que determina cuál es el índice de la       palabra p cuyo contexto es C.

        """

        st.image("Word2Vec-Training-Models.png")


        """ Ambos modelos aprenden a representar las palabras basados en un contexto que se define como una ventana alrededor de la palabra que         indica cuántas palabras alrededor voy a tener en cuenta para la predicción. 
        La ventaja de estos modelos es que permiten aprender estos *embedings* de forma eficiente (poca complejidad espacial y temporal)        pudiendo ser aplicado sobre corpus de billones de palabras.
        """

        """ Veamos ahora un ejemplo de como entrenar nuestro propios embedings usando Gensim y el algoritmo Word2Vec y la arquitectura skip-gram        :"""

        st.code(
        """
        
        from nltk.corpus import cess_esp
        from gensim.models import Word2Vec
        import multiprocessing

        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        import plotly.express as px
        import plotly.graph_objects as go

        sentences = cess_esp.sents()

        EMB_DIM = 300
        multiprocessing
        numb_of_senteces = 200

        w2w = Word2Vec(sentences,size = EMB_DIM, window = 5,min_count = 5,iter = 10 , workers= multiprocessing.cpu_count())

        word_vectors = w2w.wv
        word_vectors.vectors.shape


        
        """)

        """
        *** GloVe ***

        El modelo de Vectores Globales para la Representación de Palabras (GloVe por sus siglas en inglés), se construye de manera supervisada a        partir de las estadísticas de coocurrencia entre las palabras de un corpus.
        Sea $X$ la matriz de coocurrencia entre las palabras en el corpus de entrenamiento, siendo $X_{ij}$ la entrada de dicha matriz, que se  corresponde con la cantidad de veces que la palabra $j$ aparece en el contexto de la palabra $i$. Sea $Xi = \sum_k X_{ik}$, la cantidad  de palabras que coocurren en el contexto de la palbra i. Finalmente, sea $P_{ij} = P(i,j) = P(j|i)/P(i)$ la probabilidad de que la       palabra $j$ ocurra en el contexto de la palabra $i$. Este valor se puede estimar con la fórmula $X_{ij}=X_i$. La relación entre dos   palabras $i,j$ puede ser analizada examinando la razón de su probabilidad de coocurrencia con varias palabras de prueba. Si la palabra k  está relacionada con $i$ y $j$, entonces la razón $P_{ik}/P_{jk}$ deberá tener un valor cercano a 1 (de la misma forma si $k$ no está    relacionada con ninguna de las dos). Por el contrario, si $k$ está relacionada con una de los dos, y no con la otra, el valor de dicha     razón deberá estar distante de 1.
        Sea $w_i,wj \in R^d$ la representación de las palabras $i,j$ en un espacio vectorial de dimensión $d$, y $w^{`}_k \in R^d$ la   reprentación en dicho espacio de la palabra $k$ cuando aparece como contexto. Entonces, una primera aproximación para obtener estas       representaciones es suponer que la razón entre las probabilidades $P_{ik}$ y $P_{jk}$, dependa funcionalmente de ellas. Esto es, siendo       $F : R^d$ X $R^d$ X $R^d \implies R$ una función:

        $$
        F(w_i,w_j,w^{`}_k) = P_{ik}/P_{jk}       (1)
        $$                                                                        

        En esta ecuación, el miembro derecho se extrae del corpus, y $F$ depende de ciertos parámetros hasta ahora no especificados. Existen    diversas opciones para escoger la forma de la función $F$. Para cumplir ciertos requisitos esperados de dicha función se logra limitar
        este conjunto de opciones.

        Primero, haciendo suposiciones de linealidad sobre la función F y basándose en la naturaleza intrínsecamente lineal de los espacios     vectoriales, se puede modificar la ecuación 1: 

        $$
        F((w_{i} - w_j)^T w^{`}_k) = P_{ik}/P_{jk} (2)
        $$                                                                             

        Nótese además que en las matrices de coocurrencia de términos, la distinción entre una palabra y una palabra de contexto es arbitraria,         y estos roles pueden ser intercambiados. Es necesario lograr que en $F$, no solo que $w$ y $w^{`}$ intercambie, sino que también lo     hagan $X$ y $X^T$ . Esta simetría puede ser alcanzada en dos pasos. Primero, requiriendo que F sea un homeomorfismo entre los grupos (R;    +) y (R>0;X), por ejemplo:

        $$
        F((w_i - w_j)^{T} w^{`}) = F(w^T_i w^{`}_k) / F(w^T_j w^{`}_k) (3)
        $$  

        Se pude satisfacer la función 2 haciendo: 

        $$
        F(w^T_i w^{`}_k) = P_{ik} =X_{ik}/X_i
        $$

        Y la solución $F$ que satisface la ecuación 3 es $F(x) = exp$, o lo que es lo mismo:

        $$
        w^T_i w^{`}_k = log(P_{ik}) = log(X_{ik}) - log(X_i)
        $$

        Luego, para lograr la deseada simetría entre $i$ y $k$, el sumando $log(X_i)$ puede ser absorvido por un error bi independiente de k y  agregársele otro error $b_k$ que dependa de $k$. Finalmente la expresión quedaría:

        $$
        w^T_i w^{`}_k + b_i + b_k = log(X_{ik})
        $$

        Para resolver la indefinición del lograritmo cuando $X_{ik} = 0$ usualmente se sustituye $log(X_{ik})$ por $log(1 +X_{ik})$.

        Los autores proponen entrenar dicho modelo con esquema de mínimos cuadrados, y para ello proponen una función de pérdida $J$ ponderada  por una función $f(X_{ij} )$:

        $$
        J =\sum_{ij}^{V} f(X_{ij})(w^{T}_i w^{`}_k + b_i + b_k - log(X_{ik}))^2
        $$

        La función $f(X_{ij})$ debe cumplir que sea no decreciente, para que no se sobrestime el peso de las coocurrencias que no son   frecuentes. Además debe tener valores relativamente pequeños también para grandes valores de $X_{ij}$, para que las coocurrencias muy     frecuentes tampoco se sobreponderen. Una función que captura estas nociones, recomendada por los autores es:

        $$f(x) = \left \{
                        \\begin{array}{lccc}
                        (x/ x_{max})^\\alpha & si & x < x_{max} & \\\
                        \\\ 1 & en  & otro & caso
                        \end{array}
                \\right.
        $$
        
        """

        """
        *** Visualización ***

        Es posible visualizar estos embedings, pero debido a sus muchas dimensiones es necesario aplicar algún algoritmo para reducir   dimensiones como **PCA**. A continuación mostramos un método que dado el modelo anterior entrenado por Gensim reduce dimensiones y        plotea los vectores de palabras como puntos en el plano donde se aprecia de cierta forma como se acercan y se alejan las palabras según        su significado. Recordar que la efectividad de estos embedings depende mucho del corpus usado, el usado para la demostración solo      contine algo más de 4000 palabras en su vocabulario, en la práctica estos embedings son entrenados con billones de palabras.

        """
        st.code( 
        """
        
        def tsne_plot(model,dim):
            "Creates and TSNE model and plots it"
            labels = []
            tokens = []

            for word in model.wv.vocab:
                tokens.append(model[word])
                labels.append(word)
        
            tsne_model = TSNE(perplexity=40, n_components=dim, init='pca', n_iter=2500, random_state=23)
            new_values = tsne_model.fit_transform(tokens)

            x = []
            y = []
            z= []
            for value in new_values:
                x.append(value[0])
                y.append(value[1])
                if dim == 3:
                    z.append(value[2])

            if dim == 3:
                fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                           mode='markers',text = labels)])
                fig.show()
            else:
                fig = px.scatter(x,y, text= labels)
                fig.show()


        

        """
        )


        """
        Para probar con los diferentes parámetros de este modelo de embeding entrenado pueden usar la opción de entrenar embeding del panel desplegablea a la izquierda y ver los diferenes  resultados obtenidos. 
        """
        """
        *** Papers y más información ***
        * Original GloVe Paper: https://nlp.stanford.edu/pubs/glove.pdf
        * Original t-SNE Paper:  http://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf
        * Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. ICLR           Workshop, 2013.
        * Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and          their compositionality. Proceedings of the 26th International Conference on Neural Information Processing Systems, 2013.
        * BERT original paper: https://arxiv.org/abs/1810.04805
        * Google it!!! 😉 👍


        """



if val == "Entrenando y visualizando embedings" :



        @st.cache
        def tsne_plot(model,dim):
            "Creates and TSNE model and plots it"
            labels = []
            tokens = []

            for word in model.wv.vocab:
                tokens.append(model[word])
                labels.append(word)
        
            tsne_model = TSNE(perplexity=40, n_components=dim, init='pca', n_iter=2500, random_state=23)
            new_values = tsne_model.fit_transform(tokens)

            x = []
            y = []
            z= []
            for value in new_values:
                x.append(value[0])
                y.append(value[1])
                if dim == 3:
                    z.append(value[2])

            if dim == 3:
                fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                           mode='markers',text = labels)])
                return fig
                # st.plotly_chart(fig)
                # fig.show()
            else:
                fig = px.scatter(x,y, text= labels)
                return fig
                # st.plotly_chart(fig)
                # fig.show()
        @st.cache(suppress_st_warning=True,allow_output_mutation=True)
        def buildModel(EMB_DIM,window):

                sentences = cess_esp.sents()
                w2w = Word2Vec(sentences[:100],size = EMB_DIM, window = window,min_count = 5,iter = 10 , workers= multiprocessing.cpu_count())

                word_vectors = w2w.wv
                word_vectors.vectors.shape

                fig = tsne_plot(w2w,2)
                return fig,w2w



        st.sidebar.header("Párametros")

        EMB_DIM = st.sidebar.number_input("Dimension del embeding:")

        window = st.sidebar.number_input("Tamaño de ventana:")

        train = st.sidebar.button("Train")


        if train:

               

                # EMB_DIM = 300
                # multiprocessing
                # numb_of_senteces = 200

                fig,w2w = buildModel(EMB_DIM,window)
                st.plotly_chart(fig)


if val == "Usando embeding preentrenados":
        """
        *** Spacy embedings ***

        Spacy es una biblioteca de python que presentan diferentes herramientas para el trabajo con texto. Incluye embedings pre-entrenados para diferentes idiomas listos para ser usados para cualquier tarea realativo a NLP.
        Estos embedings son muy sencillos de usar a continuación mostramos unos ejemplos.

        """
        st.code("""
        
        import spacy

        nlp = spacy.load("es_core_news_sm")

        nlp("perro").vector

        
        """)

        " La última línea devuelve un vector de dimensión 300 con la representación del la palabra perro. Este embeding no es tan bueno por lo que ahora se presentará otro usando GloVe que obtiene mejores resultados"

        """
        *** GleVe ***

        Existen diferentes embedings pre-entrenados con GloVe, a continuación le mostramos el uso de uno para el idioma inglés que tiene un tamaño de 822 megas que se puede descargar de esta dirección http://nlp.stanford.edu/data/glove.6B.zip. También pueden encontrar versiones de mayor tamaño en este link https://nlp.stanford.edu/projects/glove/.

        Los embedings están en forma de txt y el formato es palabra n númenros separados por espacio, donde n nos dice la dimensión del embeding. Lo siguiente es un ejemplo:
        """
        st.code("business 0.023693 0.13316 0.023131 ...")

        """
        Pasaremos ahora a cargar estos embedings, para esto utilizaremos un diccionario para mapear cada palabra a su respectivo vector. Luego crearemos una función para poder buscar las palabras más similares a una determinada y también mostrar algunas aritmétricas entre estos vectores donde se pueden ver diferentes resultados interesantes que nos muestran el poder de GloVe.
        """

        st.code("""
        def find_closest_embeddings(embedding):
                return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))
        
        
        embeddings_dict = {}
        with open("glove.6B.50d.txt", 'r') as f:
                for line in f:
                        values = line.split()
                        word = values[0]
                        vector = np.asarray(values[1:], "float32")
                        embeddings_dict[word] = vector
        

        """)
        """
        A continuación vamos a poder realizar algunos experimentos usando el código anterior donde podemos hacer lagunas opercaiones sobre los vectores y ver para una palabra específica cuales son más cercanas.
        
        """

        @st.cache
        def find_closest_embeddings(embedding,embeddings_dict):
                return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))
        
        @st.cache
        def load_embeding(file="glove.6B.50d.txt" ):
                embeddings_dict = {}
                with open(file, 'r', encoding='utf-8') as f:
                        for line in f:
                                values = line.split()
                                word = values[0]
                                vector = np.asarray(values[1:], "float32")
                                embeddings_dict[word] = vector
                        return embeddings_dict
        
        @st.cache
        def plot_GLoVe(embeddings_dict,numb_of_words=1000):
                tsne = TSNE(n_components=2,)


                words =  list(embeddings_dict.keys())
                vectors = [embeddings_dict[word] for word in words]

                val = tsne.fit_transform(vectors[:numb_of_words])

                fig = px.scatter(val[:,0],val[:,1], text= words[:numb_of_words])
                return fig

        embedding =  load_embeding()

        fig = plot_GLoVe(embedding)
        st.plotly_chart(fig)


        st.info("Embeding loaded!")


        word = st.sidebar.text_input("Escriba una palabra:")
        

        if word:

                results = find_closest_embeddings(embedding[word],embedding )

                if results:
                        "Más similares a " + word +" son: "
                        results[:6]
        
        ##################################################################


        ###################################################################