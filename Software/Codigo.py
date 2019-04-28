import pandas as pd
import numpy as np
import time
from scipy.spatial import KDTree
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedKFold

np.random.seed(73)


"""
Lectura de datos
"""
df = pd.read_csv('doc/texture.csv')
data = df.values
datos = data[:,1:]


"""
División de los datos en 5 partes iguales conservando la
proporción 80%-20%. Se guardarán en las listas train y test
"""
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=73)
train = []
test = []

x = datos[:,:datos.shape[1]-1]
y = datos[:,datos.shape[1]-1]

for train_index, test_index in skf.split(x, y):
   x_train, y_train = x[train_index], y[train_index]
   x_test, y_test   = x[test_index], y[test_index]

   train.append([x_train, y_train])
   test.append([x_test, y_test])


"""
Función de evaluación para sacar los 4 parámetros de las tablas
Devuelve los porcentajes y el tiempo que ha tardado
"""
def evaluar(w, atributos, clases, antes):
    prod = (atributos * w)[:, w > 0.2]
    
    arbol = cKDTree(prod)
    vecinos = arbol.query(prod, k=2)[1][:, 1]
    
    _clas = np.mean(clases[vecinos] == clases)
    _red = np.mean(w < 0.2)
    despues = time.time()
    
    return _clas*100, _red*100, (_clas*0.5 + _red*0.5)*100, despues - antes


"""
Función obtenerGanancia para comparar los valores de la BL
Devuelve los porcentaje y el tiempo que ha tardado
"""
def obtenerGanancia(w, atributos, clases):
    prod = (atributos * w)[:, w > 0.2]

    arbol = cKDTree(prod)
    vecinos = arbol.query(prod, k=2)[1][:, 1]
    
    _clas = np.mean(clases[vecinos] == clases)
    _red = np.mean(w < 0.2)
    
    return (_clas*0.5 + _red*0.5)



"""
Algoritmo de Búsqueda Local
Ejecutado 5 veces para cada uno de los datos anteriores
"""
maxEval = 15000
maxVec = 20
resultadosBL = []

for ejecucion in range(5):
    antes = time.time()
    w = np.random.random_sample((x.shape[1],))
    ganancia = obtenerGanancia(w, test[ejecucion][0], test[ejecucion][1])
    nEval = 0
    nAtrib = 0
    fin = False
    
    while nEval < maxEval and nAtrib < (maxVec*x.shape[1]) and fin == False:
        wActual = np.copy(w)
        
        for posNuevo in np.random.permutation(x.shape[1]):
            w[posNuevo] += np.random.normal(loc=0, scale=(0.3**2))
            w[posNuevo] = np.clip(w[posNuevo], 0, 1)
            gananciaNueva = obtenerGanancia(w, train[ejecucion][0], train[ejecucion][1])
            nEval += 1
        
            if gananciaNueva > ganancia:
                ganancia = gananciaNueva
                nAtrib = 0
                break
            else:
                nAtrib+=1
                w[posNuevo] = wActual[posNuevo]
                
            if nEval > maxEval or nAtrib > maxVec*x.shape[1]:
                fin = True
                break

    resultadosBL.append(evaluar(w, test[ejecucion][0], test[ejecucion][1], antes))


"""
Algoritmo RELIEF
Ejecutado 5 veces para cada uno de los datos anteriores
"""
w = np.zeros((5,datos.shape[1]-1))
resultadosRELIEF = []

for ejecucion in range(5):
    antes = time.time()
    
    atributos = train[ejecucion][0]
    clases = train[ejecucion][1]

    for i in range(atributos.shape[0]):
        amigos = atributos[ clases == clases[i] ]
        enemigos = atributos[ clases != clases[i] ]

        tree_am = KDTree(amigos)
        indiceAmigo = tree_am.query(atributos[i].reshape(1,-1), k=2)[1][0,1]

        tree_en = KDTree(enemigos)
        indiceEnemigo = tree_en.query(atributos[i].reshape(1,-1), k=1)[1][0]

        amigo = amigos[indiceAmigo]
        enemigo = enemigos[indiceEnemigo]

        w[ejecucion] += abs(atributos[i]-enemigo) - abs(atributos[i]-amigo)

    w[ejecucion] = w[ejecucion]/max(w[ejecucion])
    w[ejecucion] = np.clip(w[ejecucion], 0, 1)

    resultadosRELIEF.append(evaluar(w[ejecucion], test[ejecucion][0], test[ejecucion][1], antes))


df = pd.DataFrame(resultadosBL)
#df = pd.DataFrame(resultadosRELIEF)
#df.to_csv('result.csv', index=False)
print(df)