import pandas as pd
import numpy as np
import time
from scipy.spatial import KDTree
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedKFold


# -----------------------------------------------------------------------------------
# Lectura de datos
# -----------------------------------------------------------------------------------
def read_data(file):
	df = pd.read_csv(file)
	data = df.values
	return data[:,1:]


# -----------------------------------------------------------------------------------
# Función de evaluación para sacar los 4 parámetros de las tablas
# Devuelve los porcentajes y el tiempo que ha tardado
# -----------------------------------------------------------------------------------
def evaluar(w, atributos, clases, antes):
    prod = (atributos * w)[:, w > 0.2]

    arbol = cKDTree(prod)
    vecinos = arbol.query(prod, k=2)[1][:, 1]

    _clas = np.mean(clases[vecinos] == clases)
    _red = np.mean(w < 0.2)
    despues = time.time()

    return _clas*100, _red*100, (_clas*0.5 + _red*0.5)*100, despues - antes


# -----------------------------------------------------------------------------------
# Función obtenerGanancia para comparar los valores de la BL
# Devuelve los porcentaje y el tiempo que ha tardado
# -----------------------------------------------------------------------------------
def obtenerGanancia(w, atributos, clases):
    prod = (atributos * w)[:, w > 0.2]

    arbol = cKDTree(prod)
    vecinos = arbol.query(prod, k=2)[1][:, 1]

    _clas = np.mean(clases[vecinos] == clases)
    _red = np.mean(w < 0.2)

    return (_clas*0.5 + _red*0.5)


# -----------------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------------
def cruceBLX(cromosoma1, cromosoma2, alpha):
    w1 = np.copy(cromosoma1)
    w2 = np.copy(cromosoma2)

    for i in range(w1.size):
        cMax = max(w1[i],w2[i])
        cMin = min(w1[i],w2[i])
        I = cMax - cMin

        w1[i] = np.random.uniform(cMin - I*alpha, cMax + I*alpha)
        w2[i] = np.random.uniform(cMin - I * alpha, cMax + I * alpha)

        w1[i] = np.clip(w1[i], 0, 1)
        w2[i] = np.clip(w2[i], 0, 1)

    return w1, w2


# -----------------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------------
def cruceAritmetico(cromosoma1, cromosoma2):
    w1 = np.copy(cromosoma1)
    w2 = np.copy(cromosoma2)

    w = []
    for i in range(w1.size):
        w.append( (w1[i] + w2[i])/2 )

    return np.array(w)


# -----------------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------------
def mutate(cromosoma):
    if np.random.random() <= 0.001:
        punto = np.random.randint(0, cromosoma.size)
        cromosoma[punto] += np.random.normal(0, 0.3)

    return cromosoma





def busquedaLocal(train, test, maxEval, maxVec, numAtributos):
	w = np.random.random_sample((numAtributos,))
    ganancia = obtenerGanancia(w, test[0], test[1])
    contEval = 0
    contAtrib = 0
    fin = False

    while contEval < maxEval and contAtrib < (maxVec*numAtributos) and fin == False:
        wActual = np.copy(w)

        for posNuevo in np.random.permutation(numAtributos):
            w[posNuevo] += np.random.normal(loc=0, scale=(0.3**2))
            w[posNuevo] = np.clip(w[posNuevo], 0, 1)
            gananciaNueva = obtenerGanancia(w, train[0], train[1])
            contEval += 1

            if gananciaNueva > ganancia:
                ganancia = gananciaNueva
                contAtrib = 0
                break
            else:
                contAtrib += 1
                w[posNuevo] = wActual[posNuevo]

            if contEval > maxEval or contAtrib > maxVec*numAtributos:
                fin = True
                break

    return w











