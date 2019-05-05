import pandas as pd
import numpy as np
import time
from scipy.spatial import KDTree
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedKFold

np.random.seed(76067801)


# -----------------------------------------------------------------------------------
# Lectura de datos
# -----------------------------------------------------------------------------------
df = pd.read_csv('doc/ionosphere.csv')
data = df.values
datos = data[:,1:]


# -----------------------------------------------------------------------------------
# División de los datos en 5 partes iguales conservando la
# proporción 80%-20%. Se guardarán en las listas train y test
# -----------------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------------
# Menu de control de la aplicación
# (Sin relación con la práctica, solo para facilizar el uso)
# -----------------------------------------------------------------------------------
print("Seleccione cómo quiere modificar los datos:")
print("1 - Búsqueda Local")
print("2 - Greedy RELIEF")
print("3 - AGG Cruce BLX")
print("4 - AGG Cruce Aritmético")
print("5 - AGE Cruce BLX")
print("6 - AGE Cruce Aritmético")
opcion = input()
resultados = []


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


def cruceAritmetico(cromosoma1, cromosoma2):
    w1 = np.copy(cromosoma1)
    w2 = np.copy(cromosoma2)

    w = []
    for i in range(w1.size):
        w.append( (w1[i] + w2[i])/2 )

    return np.array(w)



def mutate(cromosoma):
    if np.random.random() <= 0.001:
        punto = np.random.randint(0, cromosoma.size)
        cromosoma[punto] += np.random.normal(0, 0.3)

    return cromosoma



# -----------------------------------------------------------------------------------
# ---------------------------- Algoritmo Búsqueda Local -----------------------------
# -----------------------------------------------------------------------------------
if opcion == '1':
    maxEval = 15000
    maxVec = 20

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

        resultados.append(evaluar(w, test[ejecucion][0], test[ejecucion][1], antes))



# -----------------------------------------------------------------------------------
# ----------------------------- Algoritmo Gredy RELIEF ------------------------------
# -----------------------------------------------------------------------------------
elif opcion == '2':
    w = np.zeros((5,datos.shape[1]-1))

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

        resultados.append(evaluar(w[ejecucion], test[ejecucion][0], test[ejecucion][1], antes))




# -----------------------------------------------------------------------------------
# ------------------ Algoritmo Genético Generacional (Cruce BLX) --------------------
# -----------------------------------------------------------------------------------
elif opcion == '3':
    maxEval = 15000
    tamPoblacion = 30

    for ejecucion in range(5):
        print(ejecucion)
        antes = time.time()
        nEval = 0
        gMaxHijo = 0
        gMinHijo = 1

        nuevaPoblacion = np.random.uniform(0, 1, size=(30, datos.shape[1] - 1))
        gMaxPadre = 0

        for i in range(tamPoblacion):
            ganancia = obtenerGanancia(nuevaPoblacion[i], train[ejecucion][0], train[ejecucion][1])
            if ganancia > gMaxPadre:
                gMaxPadre = ganancia
                posMaxPadre = i
                w = np.copy(nuevaPoblacion[i])

        while nEval < maxEval:
            poblacion = np.copy(nuevaPoblacion)

            # Seleccionar P(t)
            nuevaPoblacion = []
            for i in range(tamPoblacion):
                nEval += 1
                index = np.random.randint(tamPoblacion, size=2)
                cromo1, cromo2 = poblacion[index]

                if obtenerGanancia(cromo1, train[ejecucion][0], train[ejecucion][1]) < \
                        obtenerGanancia(cromo2, train[ejecucion][0], train[ejecucion][1]):
                    nuevaPoblacion.append(cromo1)
                else:
                    nuevaPoblacion.append(cromo2)

            nuevaPoblacion = np.array(nuevaPoblacion)


            for p in range(0,int(tamPoblacion*0.7)-1,2):
                # Cruce P(t)
                cromo1, cromo2 = cruceBLX(nuevaPoblacion[p], nuevaPoblacion[p+1], 0.3)
                ganancia1 = obtenerGanancia(cromo1, train[ejecucion][0], train[ejecucion][1])
                ganancia2 = obtenerGanancia(cromo2, train[ejecucion][0], train[ejecucion][1])

                # Evaluar P(t)
                if ganancia1 > ganancia2:
                    cromo1 = (mutate(cromo1))
                    nuevaPoblacion[p] = np.copy(cromo1)

                    if ganancia1 > gMaxHijo:
                        w = np.copy(cromo1)
                        gMaxHijo = ganancia1
                        posMaxHijo = p

                    elif ganancia1 < gMinHijo:
                        gMinHijo = ganancia1
                        posMinHijo = p


                else:
                    cromo2 = (mutate(cromo2))
                    nuevaPoblacion[p+1] = np.copy(cromo2)

                    if ganancia2 > gMaxHijo:
                        w = np.copy(cromo2)
                        gMaxHijo = ganancia2
                        posMaxHijo = p+1

                    elif ganancia2 < gMinHijo:
                        gMinHijo = ganancia2
                        posMinHijo = p+1


            if gMinHijo < gMaxPadre:
                nuevaPoblacion[posMinHijo] = poblacion[posMaxPadre]
            else:
                posMaxPadre = posMinHijo

        resultados.append(evaluar(w, test[ejecucion][0], test[ejecucion][1], antes))


# ------------------------------------------------------------------------------------
# --------------- Algoritmo Genético Generacional (Cruce Aritmético) -----------------
# ------------------------------------------------------------------------------------
elif opcion == '4':
    maxEval = 15000
    tamPoblacion = 30

    for ejecucion in range(5):
        print(ejecucion)
        antes = time.time()
        nEval = 0
        gMaxHijo = 0
        gMinHijo = 1

        nuevaPoblacion = np.random.uniform(0, 1, size=(30, datos.shape[1] - 1))
        gMaxPadre = 0

        for i in range(tamPoblacion):
            ganancia = obtenerGanancia(nuevaPoblacion[i], train[ejecucion][0], train[ejecucion][1])
            if ganancia > gMaxPadre:
                gMaxPadre = ganancia
                posMaxPadre = i
                w = np.copy(nuevaPoblacion[i])

        while nEval < maxEval:
            poblacion = np.copy(nuevaPoblacion)

            nuevaPoblacion = []
            for p in range(int(tamPoblacion * 1.4)):
                nEval += 1
                # Cruce P(t)
                cromo1 = cruceAritmetico(poblacion[np.random.randint(tamPoblacion)], poblacion[np.random.randint(tamPoblacion)])
                cromo2 = cruceAritmetico(poblacion[np.random.randint(tamPoblacion)], poblacion[np.random.randint(tamPoblacion)])
                ganancia1 = obtenerGanancia(cromo1, train[ejecucion][0], train[ejecucion][1])
                ganancia2 = obtenerGanancia(cromo2, train[ejecucion][0], train[ejecucion][1])

                # Evaluar P(t)
                if ganancia1 > ganancia2:
                    cromo1 = (mutate(cromo1))
                    nuevaPoblacion.append(cromo1)

                    if ganancia1 > gMaxHijo:
                        w = np.copy(cromo1)
                        gMaxHijo = ganancia1
                        posMaxHijo = p

                    elif ganancia1 < gMinHijo:
                        gMinHijo = ganancia1
                        posMinHijo = p


                else:
                    cromo2 = (mutate(cromo2))
                    nuevaPoblacion.append(cromo2)

                    if ganancia2 > gMaxHijo:
                        w = np.copy(cromo2)
                        gMaxHijo = ganancia2
                        posMaxHijo = p

                    elif ganancia2 < gMinHijo:
                        gMinHijo = ganancia2
                        posMinHijo = p

            nuevaPoblacion = np.array(nuevaPoblacion)
            print(w)
            if gMinHijo < gMaxPadre:
                nuevaPoblacion[posMinHijo] = poblacion[posMaxPadre]
            else:
                posMaxPadre = posMinHijo

        resultados.append(evaluar(w, test[ejecucion][0], test[ejecucion][1], antes))




# -----------------------------------------------------------------------------------
# ------------------ Algoritmo Genético Estacionario (Cruce BLX) --------------------
# -----------------------------------------------------------------------------------
elif opcion == '5':
    maxEval = 15000
    tamPoblacion = 30

    for ejecucion in range(5):
        print(ejecucion)
        antes = time.time()
        nEval = 0
        gananciaMax = 0
        gMinPadre = 0
        gMinPadre2 = 0
        gPosPadre = 0
        gPosPadre2 = 0

        poblacion = np.random.uniform(0, 1, size=(30, datos.shape[1] - 1))


        while nEval < maxEval:

            for i in range(tamPoblacion):
                nEval += 1
                ganancia = obtenerGanancia(poblacion[i], train[ejecucion][0], train[ejecucion][1])

                if ganancia > gananciaMax:
                    gananciaMax = ganancia
                    w = np.copy(poblacion[i])

                if ganancia > gMinPadre2:
                    gMinPadre = gMinPadre2
                    gPosPadre = gPosPadre2
                    gMinPadre2 = ganancia
                    gPosPadre2 = i

                elif ganancia > gMinPadre:
                    gMinPadre = ganancia
                    gPosPadre = i


            # Seleccionar P(t)
            nuevaPoblacion = []
            for i in range(2):
                index = np.random.randint(tamPoblacion, size=2)
                cromo1, cromo2 = poblacion[index]

                if obtenerGanancia(cromo1, train[ejecucion][0], train[ejecucion][1]) < \
                        obtenerGanancia(cromo2, train[ejecucion][0], train[ejecucion][1]):
                    nuevaPoblacion.append(cromo1)
                else:
                    nuevaPoblacion.append(cromo2)

            nuevaPoblacion = np.array(nuevaPoblacion)


            # Cruce P(t)
            cromo1, cromo2 = cruceBLX(nuevaPoblacion[0], nuevaPoblacion[1], 0.3)
            ganancia1 = obtenerGanancia(cromo1, train[ejecucion][0], train[ejecucion][1])
            ganancia2 = obtenerGanancia(cromo2, train[ejecucion][0], train[ejecucion][1])

            if ganancia1 > gMinPadre2:
                poblacion[gPosPadre2] = np.copy(mutate(cromo1))
                gMinPadre = gMinPadre2
                gMinPadre2 = ganancia1

            elif ganancia1 > gMinPadre:
                poblacion[gPosPadre] = np.copy(mutate(cromo1))
                gMinPadre = ganancia1

            if ganancia2 > gMinPadre2:
                poblacion[gPosPadre2] = np.copy(mutate(cromo2))
                gMinPadre = gMinPadre2
                gMinPadre2 = ganancia2

            elif ganancia2 > gMinPadre:
                poblacion[gPosPadre] = np.copy(mutate(cromo2))
                gMinPadre = ganancia2



        for i in range(tamPoblacion):
            ganancia = obtenerGanancia(poblacion[i], train[ejecucion][0], train[ejecucion][1])

            if ganancia > gananciaMax:
                gananciaMax = ganancia
                w = np.copy(poblacion[i])

        resultados.append(evaluar(w, test[ejecucion][0], test[ejecucion][1], antes))




# ------------------------------------------------------------------------------------
# --------------- Algoritmo Genético Estacionario (Cruce Aritmético) -----------------
# ------------------------------------------------------------------------------------
elif opcion == '6':
    maxEval = 15000
    tamPoblacion = 30

    for ejecucion in range(5):
        print(ejecucion)
        antes = time.time()
        nEval = 0
        gananciaMax = 0
        gMinPadre = 0
        gMinPadre2 = 0
        gPosPadre = 0
        gPosPadre2 = 0

        poblacion = np.random.uniform(0, 1, size=(30, datos.shape[1] - 1))

        while nEval < maxEval:

            for i in range(tamPoblacion):
                nEval += 1
                ganancia = obtenerGanancia(poblacion[i], train[ejecucion][0], train[ejecucion][1])

                if ganancia > gananciaMax:
                    gananciaMax = ganancia
                    w = np.copy(poblacion[i])

                if ganancia > gMinPadre2:
                    gMinPadre = gMinPadre2
                    gPosPadre = gPosPadre2
                    gMinPadre2 = ganancia
                    gPosPadre2 = i

                elif ganancia > gMinPadre:
                    gMinPadre = ganancia
                    gPosPadre = i

            # Seleccionar P(t)
            nuevaPoblacion = []
            for i in range(4):
                index = np.random.randint(tamPoblacion, size=2)
                cromo1, cromo2 = poblacion[index]

                if obtenerGanancia(cromo1, train[ejecucion][0], train[ejecucion][1]) < \
                        obtenerGanancia(cromo2, train[ejecucion][0], train[ejecucion][1]):
                    nuevaPoblacion.append(cromo1)
                else:
                    nuevaPoblacion.append(cromo2)

            nuevaPoblacion = np.array(nuevaPoblacion)

            # Cruce P(t)
            cromo1 = cruceAritmetico(nuevaPoblacion[0], nuevaPoblacion[1])
            cromo2 = cruceAritmetico(nuevaPoblacion[2], nuevaPoblacion[3])
            ganancia1 = obtenerGanancia(cromo1, train[ejecucion][0], train[ejecucion][1])
            ganancia2 = obtenerGanancia(cromo2, train[ejecucion][0], train[ejecucion][1])

            if ganancia1 > gMinPadre2:
                poblacion[gPosPadre2] = np.copy(mutate(cromo1))
                gMinPadre = gMinPadre2
                gMinPadre2 = ganancia1

            elif ganancia1 > gMinPadre:
                poblacion[gPosPadre] = np.copy(mutate(cromo1))
                gMinPadre = ganancia1

            if ganancia2 > gMinPadre2:
                poblacion[gPosPadre2] = np.copy(mutate(cromo2))
                gMinPadre = gMinPadre2
                gMinPadre2 = ganancia2

            elif ganancia2 > gMinPadre:
                poblacion[gPosPadre] = np.copy(mutate(cromo2))
                gMinPadre = ganancia2

        for i in range(tamPoblacion):
            ganancia = obtenerGanancia(poblacion[i], train[ejecucion][0], train[ejecucion][1])

            if ganancia > gananciaMax:
                gananciaMax = ganancia
                w = np.copy(poblacion[i])

        resultados.append(evaluar(w, test[ejecucion][0], test[ejecucion][1], antes))




else:
    print("Opción incorrecta")


df = pd.DataFrame(resultados)
#df.to_csv('result.csv', index=False)
print(df)