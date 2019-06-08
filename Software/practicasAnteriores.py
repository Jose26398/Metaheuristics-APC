import pandas as pd
import numpy as np
import time
import random
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





def busquedaLocal(train, maxEval, maxVec, numAtributos):
    w = np.random.random_sample((numAtributos,))
    ganancia = obtenerGanancia(w, train[0], train[1])
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



def greedyRELIEF(train, numAtributos):
    w = np.zeros(numAtributos)

    atributos = train[0]
    clases = train[1]

    for i in range(atributos.shape[0]):
        amigos = atributos[clases == clases[i]]
        enemigos = atributos[clases != clases[i]]

        tree_am = KDTree(amigos)
        indiceAmigo = tree_am.query(atributos[i].reshape(1, -1), k=2)[1][0, 1]

        tree_en = KDTree(enemigos)
        indiceEnemigo = tree_en.query(atributos[i].reshape(1, -1), k=1)[1][0]

        amigo = amigos[indiceAmigo]
        enemigo = enemigos[indiceEnemigo]

        w += abs(atributos[i] - enemigo) - abs(atributos[i] - amigo)

    w = w / max(w)
    return np.clip(w, 0, 1)



def geneticoGeneracionalBLX(train, maxEval, tamPoblacion, numAtributos):
    nEval = 0

    poblacion = np.random.uniform(0, 1, size=(30, numAtributos))

    ganancias = []
    for i in range(tamPoblacion):
        ganancias.append([i, obtenerGanancia(poblacion[i], train[0], train[1])])

    ganancias = np.array(ganancias)
    ganancias.view('i8,i8').sort(order=['f1'], axis=0)
    ganancias = ganancias[::-1]

    while nEval < maxEval:
        # Seleccionar P(t) torneo binario
        padres = []
        gananciasPadres = []
        for i in range(tamPoblacion):
            nEval += 1
            index = np.random.randint(tamPoblacion, size=2)
            cromo1, cromo2 = poblacion[index]
            ganancia1 = obtenerGanancia(cromo1, train[0], train[1])
            ganancia2 = obtenerGanancia(cromo2, train[0], train[1])

            if ganancia1 > ganancia2:
                padres.append(cromo1)
                gananciasPadres.append([i, ganancia1])
            else:
                padres.append(cromo2)
                gananciasPadres.append([i, ganancia2])

        padres = np.array(padres)
        gananciasPadres = np.array(gananciasPadres)

        # Cruce P(t) con BLX
        hijos = []
        gananciasHijos = []
        for p in range(0, int(tamPoblacion * 0.7), 2):
            hijo1, hijo2 = cruceBLX(padres[p], padres[p + 1], 0.3)
            hijos.append(mutate(hijo1))
            hijos.append(mutate(hijo2))
            gananciasHijos.append([p, obtenerGanancia(hijo1, train[0], train[1])])
            gananciasHijos.append([p + 1, obtenerGanancia(hijo2, train[0], train[1])])

        for p in range(int(tamPoblacion * 0.7), tamPoblacion):
            hijos.append(padres[p])
            gananciasHijos.append(gananciasPadres[p])

        # Elitismo
        hijos = np.array(hijos)
        gananciasHijos = np.array(gananciasHijos)
        gananciasHijos.view('i8,i8').sort(order=['f1'], axis=0)

        if ganancias[0][1] > gananciasHijos[0][1]:
            hijos[int(gananciasHijos[0][0])] = np.copy(poblacion[int(ganancias[0][0])])
            gananciasHijos[0][1] = np.copy(ganancias[0][1])

        poblacion = np.copy(hijos)
        ganancias = np.copy(gananciasHijos)
        ganancias.view('i8,i8').sort(order=['f1'], axis=0)
        ganancias = ganancias[::-1]

    return poblacion[int(ganancias[0][0])]


def geneticoGeneracionalA(train, maxEval, tamPoblacion, numAtributos):
    nEval = 0

    poblacion = np.random.uniform(0, 1, size=(30, numAtributos))

    ganancias = []
    for i in range(tamPoblacion):
        ganancias.append([i, obtenerGanancia(poblacion[i], train[0], train[1])])

    ganancias = np.array(ganancias)
    ganancias.view('i8,i8').sort(order=['f1'], axis=0)
    ganancias = ganancias[::-1]

    while nEval < maxEval:
        # Seleccionar P(t) torneo binario
        padres = []
        gananciasPadres = []
        for i in range(tamPoblacion * 2):
            nEval += 1
            index = np.random.randint(tamPoblacion, size=2)
            cromo1, cromo2 = poblacion[index]
            ganancia1 = obtenerGanancia(cromo1, train[0], train[1])
            ganancia2 = obtenerGanancia(cromo2, train[0], train[1])

            if ganancia1 > ganancia2:
                padres.append(cromo1)
                gananciasPadres.append([i, ganancia1])
            else:
                padres.append(cromo2)
                gananciasPadres.append([i, ganancia2])

        padres = np.array(padres)
        gananciasPadres = np.array(gananciasPadres)

        # Cruce P(t) aritmetico
        hijos = []
        gananciasHijos = []
        aux = 0
        for p in range(int(tamPoblacion * 0.7)):
            hijo1 = cruceAritmetico(padres[aux], padres[aux + 1])
            hijos.append(mutate(hijo1))
            gananciasHijos.append([p, obtenerGanancia(hijo1, train[0], train[1])])
            aux += 2

        for p in range(int(tamPoblacion * 0.7), tamPoblacion):
            hijos.append(padres[p])
            gananciasHijos.append(gananciasPadres[p])

        # Elitismo
        hijos = np.array(hijos)
        gananciasHijos = np.array(gananciasHijos)
        gananciasHijos.view('i8,i8').sort(order=['f1'], axis=0)

        if ganancias[0][1] > gananciasHijos[0][1]:
            hijos[int(gananciasHijos[0][0])] = np.copy(poblacion[int(ganancias[0][0])])
            gananciasHijos[0][1] = np.copy(ganancias[0][1])

        poblacion = np.copy(hijos)
        ganancias = np.copy(gananciasHijos)
        ganancias.view('i8,i8').sort(order=['f1'], axis=0)
        ganancias = ganancias[::-1]

    return poblacion[int(ganancias[0][0])]



def geneticoEstacionarioBLX(train, maxEval, tamPoblacion, numAtributos):
    nEval = 0

    poblacion = np.random.uniform(0, 1, size=(30, numAtributos))

    ganancias = []
    for i in range(tamPoblacion):
        ganancias.append([i, obtenerGanancia(poblacion[i], train[0], train[1])])

    ganancias = np.array(ganancias)
    ganancias.view('i8,i8').sort(order=['f1'], axis=0)

    while nEval < maxEval:
        # Seleccionar P(t) torneo binario
        padres = []
        gananciasPadres = []
        for i in range(2):
            nEval += 1
            index = np.random.randint(tamPoblacion, size=2)
            cromo1, cromo2 = poblacion[index]
            ganancia1 = obtenerGanancia(cromo1, train[0], train[1])
            ganancia2 = obtenerGanancia(cromo2, train[0], train[1])

            if ganancia1 > ganancia2:
                padres.append(cromo1)
                gananciasPadres.append([i, ganancia1])
            else:
                padres.append(cromo2)
                gananciasPadres.append([i, ganancia2])

        padres = np.array(padres)
        gananciasPadres = np.array(gananciasPadres)

        # Cruce P(t) con BLX
        hijos = []
        gananciasHijos = []
        hijo1, hijo2 = cruceBLX(padres[0], padres[1], 0.3)
        hijos.append(mutate(hijo1))
        hijos.append(mutate(hijo2))
        gananciasHijos.append([0, obtenerGanancia(hijo1, train[0], train[1])])
        gananciasHijos.append([1, obtenerGanancia(hijo2, train[0], train[1])])

        hijos = np.array(hijos)
        gananciasHijos = np.array(gananciasHijos)

        if gananciasHijos[0][1] >= gananciasHijos[1][1]:
            mejor = gananciasHijos[0]
            peor = gananciasHijos[1]
        else:
            mejor = gananciasHijos[1]
            peor = gananciasHijos[0]

        if mejor[1] > ganancias[1][1]:
            poblacion[int(ganancias[1][0])] = hijos[int(mejor[0])]
            ganancias[1] = mejor
            if peor[1] > ganancias[0][1]:
                poblacion[int(ganancias[0][0])] = hijos[int(peor[0])]
                ganancias[0] = peor
            elif mejor[1] > ganancias[0][1]:
                poblacion[int(ganancias[0][0])] = hijos[int(mejor[0])]
                ganancias[0] = peor

        ganancias.view('i8,i8').sort(order=['f1'], axis=0)

    return poblacion[int(ganancias[0][0])]



def geneticoEstacionarioBLX(train, maxEval, tamPoblacion, numAtributos):
    nEval = 0

    poblacion = np.random.uniform(0, 1, size=(30, numAtributos))

    ganancias = []
    for i in range(tamPoblacion):
        ganancias.append([i, obtenerGanancia(poblacion[i], train[0], train[1])])

    ganancias = np.array(ganancias)
    ganancias.view('i8,i8').sort(order=['f1'], axis=0)

    while nEval < maxEval:
        # Seleccionar P(t) torneo binario
        padres = []
        gananciasPadres = []
        for i in range(2):
            nEval += 1
            index = np.random.randint(tamPoblacion, size=2)
            cromo1, cromo2 = poblacion[index]
            ganancia1 = obtenerGanancia(cromo1, train[0], train[1])
            ganancia2 = obtenerGanancia(cromo2, train[0], train[1])

            if ganancia1 > ganancia2:
                padres.append(cromo1)
                gananciasPadres.append([i, ganancia1])
            else:
                padres.append(cromo2)
                gananciasPadres.append([i, ganancia2])

        padres = np.array(padres)
        gananciasPadres = np.array(gananciasPadres)

        # Cruce P(t) con BLX
        hijos = []
        gananciasHijos = []
        hijo1, hijo2 = cruceBLX(padres[0], padres[1], 0.3)
        hijos.append(mutate(hijo1))
        hijos.append(mutate(hijo2))
        gananciasHijos.append([0, obtenerGanancia(hijo1, train[0], train[1])])
        gananciasHijos.append([1, obtenerGanancia(hijo2, train[0], train[1])])

        hijos = np.array(hijos)
        gananciasHijos = np.array(gananciasHijos)

        if gananciasHijos[0][1] >= gananciasHijos[1][1]:
            mejor = gananciasHijos[0]
            peor = gananciasHijos[1]
        else:
            mejor = gananciasHijos[1]
            peor = gananciasHijos[0]

        if mejor[1] > ganancias[1][1]:
            poblacion[int(ganancias[1][0])] = hijos[int(mejor[0])]
            ganancias[1] = mejor
            if peor[1] > ganancias[0][1]:
                poblacion[int(ganancias[0][0])] = hijos[int(peor[0])]
                ganancias[0] = peor
            elif mejor[1] > ganancias[0][1]:
                poblacion[int(ganancias[0][0])] = hijos[int(mejor[0])]
                ganancias[0] = peor

        ganancias.view('i8,i8').sort(order=['f1'], axis=0)

    return poblacion[int(ganancias[0][0])]



def geneticoEstacionarioA(train, maxEval, tamPoblacion, numAtributos):
    nEval = 0

    poblacion = np.random.uniform(0, 1, size=(30, numAtributos))

    ganancias = []
    for i in range(tamPoblacion):
        ganancias.append([i, obtenerGanancia(poblacion[i], train[0], train[1])])

    ganancias = np.array(ganancias)
    ganancias.view('i8,i8').sort(order=['f1'], axis=0)

    while nEval < maxEval:
        # Seleccionar P(t) torneo binario
        padres = []
        gananciasPadres = []
        for i in range(2):
            nEval += 1
            index = np.random.randint(tamPoblacion, size=2)
            cromo1, cromo2 = poblacion[index]
            ganancia1 = obtenerGanancia(cromo1, train[0], train[1])
            ganancia2 = obtenerGanancia(cromo2, train[0], train[1])

            if ganancia1 > ganancia2:
                padres.append(cromo1)
                gananciasPadres.append([i, ganancia1])
            else:
                padres.append(cromo2)
                gananciasPadres.append([i, ganancia2])

        padres = np.array(padres)
        gananciasPadres = np.array(gananciasPadres)

        # Cruce P(t) con BLX
        hijos = []
        gananciasHijos = []
        hijo1, hijo2 = cruceBLX(padres[0], padres[1], 0.3)
        hijos.append(mutate(hijo1))
        hijos.append(mutate(hijo2))
        gananciasHijos.append([0, obtenerGanancia(hijo1, train[0], train[1])])
        gananciasHijos.append([1, obtenerGanancia(hijo2, train[0], train[1])])

        hijos = np.array(hijos)
        gananciasHijos = np.array(gananciasHijos)

        if gananciasHijos[0][1] >= gananciasHijos[1][1]:
            mejor = gananciasHijos[0]
            peor = gananciasHijos[1]
        else:
            mejor = gananciasHijos[1]
            peor = gananciasHijos[0]

        if mejor[1] > ganancias[1][1]:
            poblacion[int(ganancias[1][0])] = hijos[int(mejor[0])]
            ganancias[1] = mejor
            if peor[1] > ganancias[0][1]:
                poblacion[int(ganancias[0][0])] = hijos[int(peor[0])]
                ganancias[0] = peor
            elif mejor[1] > ganancias[0][1]:
                poblacion[int(ganancias[0][0])] = hijos[int(mejor[0])]
                ganancias[0] = peor

        ganancias.view('i8,i8').sort(order=['f1'], axis=0)

    return poblacion[int(ganancias[0][0])]



def memetico10_10(train, maxEval, tamPoblacion, numAtributos):
    nEval = 0
    nGeneraciones = 0

    poblacion = np.random.uniform(0, 1, size=(30, numAtributos))

    ganancias = []
    for i in range(tamPoblacion):
        ganancias.append([i, obtenerGanancia(poblacion[i], train[0], train[1])])

    ganancias = np.array(ganancias)
    ganancias.view('i8,i8').sort(order=['f1'], axis=0)
    ganancias = ganancias[::-1]

    while nEval < maxEval:
        nGeneraciones += 1

        # Seleccionar P(t) torneo binario
        padres = []
        gananciasPadres = []
        for i in range(tamPoblacion):
            nEval += 1
            index = np.random.randint(tamPoblacion, size=2)
            cromo1, cromo2 = poblacion[index]
            ganancia1 = obtenerGanancia(cromo1, train[0], train[1])
            ganancia2 = obtenerGanancia(cromo2, train[0], train[1])

            if ganancia1 > ganancia2:
                padres.append(cromo1)
                gananciasPadres.append([i, ganancia1])
            else:
                padres.append(cromo2)
                gananciasPadres.append([i, ganancia2])

        padres = np.array(padres)
        gananciasPadres = np.array(gananciasPadres)

        # Cruce P(t) con BLX
        hijos = []
        gananciasHijos = []
        for p in range(0, int(tamPoblacion * 0.7), 2):
            hijo1, hijo2 = cruceBLX(padres[p], padres[p + 1], 0.3)
            hijos.append(mutate(hijo1))
            hijos.append(mutate(hijo2))
            gananciasHijos.append([p, obtenerGanancia(hijo1, train[0], train[1])])
            gananciasHijos.append([p + 1, obtenerGanancia(hijo2, train[0], train[1])])

        for p in range(int(tamPoblacion * 0.7), tamPoblacion):
            hijos.append(padres[p])
            gananciasHijos.append(gananciasPadres[p])

        hijos = np.array(hijos)
        gananciasHijos = np.array(gananciasHijos)
        gananciasHijos.view('i8,i8').sort(order=['f1'], axis=0)

        # Busqueda Local
        if nGeneraciones % 10 == 0:
            for i in range(tamPoblacion):
                nAtrib = 0
                fin = False
                gananciaAntigua = obtenerGanancia(hijos[i], train[0], train[1])

                while nAtrib < (2 * numAtributos) and fin == False:
                    hijosBL = np.copy(hijos)
                    nEval += 1

                    for posNuevo in np.random.permutation(numAtributos):
                        hijosBL[i][posNuevo] += np.random.normal(loc=0, scale=(0.3 ** 2))
                        hijosBL[i][posNuevo] = np.clip(hijos[i][posNuevo], 0, 1)
                        gananciaNueva = obtenerGanancia(hijosBL[i], train[0], train[1])

                        if gananciaNueva > gananciaAntigua:
                            gananciaAntigua = gananciaNueva
                            nAtrib = 0
                            break
                        else:
                            nAtrib += 1
                            hijos[i][posNuevo] = np.copy(hijosBL[i][posNuevo])

                        if nAtrib > 2 * numAtributos:
                            fin = True
                            break

        # Elitismo
        if ganancias[0][1] > gananciasHijos[0][1]:
            hijos[int(gananciasHijos[0][0])] = np.copy(poblacion[int(ganancias[0][0])])
            gananciasHijos[0][1] = np.copy(ganancias[0][1])

        poblacion = np.copy(hijos)
        ganancias = np.copy(gananciasHijos)
        ganancias.view('i8,i8').sort(order=['f1'], axis=0)
        ganancias = ganancias[::-1]

    return poblacion[int(ganancias[0][0])]



def memetico10_01(train, maxEval, tamPoblacion, numAtributos):
    nEval = 0
    nGeneraciones = 0

    poblacion = np.random.uniform(0, 1, size=(30, numAtributos))

    ganancias = []
    for i in range(tamPoblacion):
        ganancias.append([i, obtenerGanancia(poblacion[i], train[0], train[1])])

    ganancias = np.array(ganancias)
    ganancias.view('i8,i8').sort(order=['f1'], axis=0)
    ganancias = ganancias[::-1]

    while nEval < maxEval:
        nGeneraciones += 1

        # Seleccionar P(t) torneo binario
        padres = []
        gananciasPadres = []
        for i in range(tamPoblacion):
            nEval += 1
            index = np.random.randint(tamPoblacion, size=2)
            cromo1, cromo2 = poblacion[index]
            ganancia1 = obtenerGanancia(cromo1, train[0], train[1])
            ganancia2 = obtenerGanancia(cromo2, train[0], train[1])

            if ganancia1 > ganancia2:
                padres.append(cromo1)
                gananciasPadres.append([i, ganancia1])
            else:
                padres.append(cromo2)
                gananciasPadres.append([i, ganancia2])

        padres = np.array(padres)
        gananciasPadres = np.array(gananciasPadres)

        # Cruce P(t) con BLX
        hijos = []
        gananciasHijos = []
        for p in range(0, int(tamPoblacion * 0.7), 2):
            hijo1, hijo2 = cruceBLX(padres[p], padres[p + 1], 0.3)
            hijos.append(mutate(hijo1))
            hijos.append(mutate(hijo2))
            gananciasHijos.append([p, obtenerGanancia(hijo1, train[0], train[1])])
            gananciasHijos.append([p + 1, obtenerGanancia(hijo2, train[0], train[1])])

        for p in range(int(tamPoblacion * 0.7), tamPoblacion):
            hijos.append(padres[p])
            gananciasHijos.append(gananciasPadres[p])

        hijos = np.array(hijos)
        gananciasHijos = np.array(gananciasHijos)
        gananciasHijos.view('i8,i8').sort(order=['f1'], axis=0)

        # Busqueda Local
        if nGeneraciones % 10 == 0:
            for j in range(int(tamPoblacion * 0.1)):
                i = np.random.randint(tamPoblacion)
                nAtrib = 0
                fin = False
                gananciaAntigua = obtenerGanancia(hijos[i], train[0], train[1])

                while nAtrib < (2 * numAtributos) and fin == False:
                    hijosBL = np.copy(hijos)
                    nEval += 1

                    for posNuevo in np.random.permutation(numAtributos):
                        hijosBL[i][posNuevo] += np.random.normal(loc=0, scale=(0.3 ** 2))
                        hijosBL[i][posNuevo] = np.clip(hijos[i][posNuevo], 0, 1)
                        gananciaNueva = obtenerGanancia(hijosBL[i], train[0], train[1])

                        if gananciaNueva > gananciaAntigua:
                            gananciaAntigua = gananciaNueva
                            nAtrib = 0
                            break
                        else:
                            nAtrib += 1
                            hijos[i][posNuevo] = np.copy(hijosBL[i][posNuevo])

                        if nAtrib > 2 * numAtributos:
                            fin = True
                            break

        # Elitismo
        if ganancias[0][1] > gananciasHijos[0][1]:
            hijos[int(gananciasHijos[0][0])] = np.copy(poblacion[int(ganancias[0][0])])
            gananciasHijos[0][1] = np.copy(ganancias[0][1])

        poblacion = np.copy(hijos)
        ganancias = np.copy(gananciasHijos)
        ganancias.view('i8,i8').sort(order=['f1'], axis=0)
        ganancias = ganancias[::-1]

    return poblacion[int(ganancias[0][0])]



def memetico10_01_mej(train, maxEval, tamPoblacion, numAtributos):
    nEval = 0
    nGeneraciones = 0

    poblacion = np.random.uniform(0, 1, size=(30, numAtributos))

    ganancias = []
    for i in range(tamPoblacion):
        ganancias.append([i, obtenerGanancia(poblacion[i], train[0], train[1])])

    ganancias = np.array(ganancias)
    ganancias.view('i8,i8').sort(order=['f1'], axis=0)
    ganancias = ganancias[::-1]

    while nEval < maxEval:
        nGeneraciones += 1

        # Seleccionar P(t) torneo binario
        padres = []
        gananciasPadres = []
        for i in range(tamPoblacion):
            nEval += 1
            index = np.random.randint(tamPoblacion, size=2)
            cromo1, cromo2 = poblacion[index]
            ganancia1 = obtenerGanancia(cromo1, train[0], train[1])
            ganancia2 = obtenerGanancia(cromo2, train[0], train[1])

            if ganancia1 > ganancia2:
                padres.append(cromo1)
                gananciasPadres.append([i, ganancia1])
            else:
                padres.append(cromo2)
                gananciasPadres.append([i, ganancia2])

        padres = np.array(padres)
        gananciasPadres = np.array(gananciasPadres)

        # Cruce P(t) con BLX
        hijos = []
        gananciasHijos = []
        for p in range(0, int(tamPoblacion * 0.7), 2):
            hijo1, hijo2 = cruceBLX(padres[p], padres[p + 1], 0.3)
            hijos.append(mutate(hijo1))
            hijos.append(mutate(hijo2))
            gananciasHijos.append([p, obtenerGanancia(hijo1, train[0], train[1])])
            gananciasHijos.append([p + 1, obtenerGanancia(hijo2, train[0], train[1])])

        for p in range(int(tamPoblacion * 0.7), tamPoblacion):
            hijos.append(padres[p])
            gananciasHijos.append(gananciasPadres[p])

        hijos = np.array(hijos)
        gananciasHijos = np.array(gananciasHijos)
        gananciasHijos.view('i8,i8').sort(order=['f1'], axis=0)
        gananciasHijos = gananciasHijos[::-1]

        # Busqueda Local
        if nGeneraciones % 10 == 0:
            for i in range(int(tamPoblacion * 0.1)):
                nAtrib = 0
                fin = False
                gananciaAntigua = obtenerGanancia(hijos[i], train[0], train[1])

                while nAtrib < (2 * numAtributos) and fin == False:
                    hijosBL = np.copy(hijos)

                    for posNuevo in np.random.permutation(numAtributos):
                        nEval += 1
                        hijosBL[i][posNuevo] += np.random.normal(loc=0, scale=(0.3 ** 2))
                        hijosBL[i][posNuevo] = np.clip(hijos[i][posNuevo], 0, 1)
                        gananciaNueva = obtenerGanancia(hijosBL[i], train[0], train[1])

                        if gananciaNueva > gananciaAntigua:
                            gananciaAntigua = gananciaNueva
                            nAtrib = 0
                            break
                        else:
                            nAtrib += 1
                            hijos[i][posNuevo] = np.copy(hijosBL[i][posNuevo])

                        if nAtrib > 2 * numAtributos:
                            fin = True
                            break

        gananciasHijos = gananciasHijos[::-1]
        # Elitismo
        if ganancias[0][1] > gananciasHijos[0][1]:
            hijos[int(gananciasHijos[0][0])] = np.copy(poblacion[int(ganancias[0][0])])
            gananciasHijos[0][1] = np.copy(ganancias[0][1])

        poblacion = np.copy(hijos)
        ganancias = np.copy(gananciasHijos)
        ganancias.view('i8,i8').sort(order=['f1'], axis=0)
        ganancias = ganancias[::-1]

    return poblacion[int(ganancias[0][0])]