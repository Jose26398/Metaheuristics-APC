from practicasAnteriores import *

np.random.seed(14)


# Lectura de datos
datos = read_data('doc/texture.csv')


# -----------------------------------------------------------------------------------
# División de los datos en 5 partes iguales conservando la
# proporción 80%-20%. Se guardarán en las listas train y test
# -----------------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=73)
train = []
test = []

X = datos[:,:datos.shape[1]-1]
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
print("7 - AM (10,1.0)")
print("8 - AM (10,0.1)")
print("9 - AM (10,0.1mej)")
opcion = input()

nEjecucuiones = 5
resultados = []


# -----------------------------------------------------------------------------------
# ---------------------------- Algoritmo Búsqueda Local -----------------------------
# -----------------------------------------------------------------------------------
if opcion == '1':
    for e in nEjecucuiones:
        antes = time.time()
        print(train[0])
        w = busquedaLocal(train[e], test[e], 15000, 20, x.shape[1])
        resultados.append(evaluar(w, test[e], test[e], antes))



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

        poblacion = np.random.uniform(0, 1, size=(30, datos.shape[1] - 1))

        ganancias = []
        for i in range(tamPoblacion):
            ganancias.append([i, obtenerGanancia(poblacion[i], train[ejecucion][0], train[ejecucion][1])])

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
                ganancia1 = obtenerGanancia(cromo1, train[ejecucion][0], train[ejecucion][1])
                ganancia2 = obtenerGanancia(cromo2, train[ejecucion][0], train[ejecucion][1])

                if  ganancia1 > ganancia2:
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
            for p in range(0,int(tamPoblacion*0.7),2):
                hijo1, hijo2 = cruceBLX(padres[p], padres[p+1], 0.3)
                hijos.append(mutate(hijo1))
                hijos.append(mutate(hijo2))
                gananciasHijos.append([p, obtenerGanancia(hijo1, train[ejecucion][0], train[ejecucion][1])])
                gananciasHijos.append([p+1, obtenerGanancia(hijo2, train[ejecucion][0], train[ejecucion][1])])

            for p in range(int(tamPoblacion * 0.7), tamPoblacion):
                hijos.append(padres[p])
                gananciasHijos.append(gananciasPadres[p])


            # Elitismo
            hijos = np.array(hijos)
            gananciasHijos = np.array(gananciasHijos)
            gananciasHijos.view('i8,i8').sort(order=['f1'], axis=0)

            if ganancias[0][1] > gananciasHijos[0][1]:
                hijos[ int(gananciasHijos[0][0]) ] = np.copy(poblacion[ int(ganancias[0][0]) ])
                gananciasHijos[0][1] = np.copy(ganancias[0][1])

            poblacion = np.copy(hijos)
            ganancias = np.copy(gananciasHijos)
            ganancias.view('i8,i8').sort(order=['f1'], axis=0)
            ganancias = ganancias[::-1]


        w = np.copy(poblacion[ int(ganancias[0][0]) ])
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

        poblacion = np.random.uniform(0, 1, size=(30, datos.shape[1] - 1))

        ganancias = []
        for i in range(tamPoblacion):
            ganancias.append([i, obtenerGanancia(poblacion[i], train[ejecucion][0], train[ejecucion][1])])

        ganancias = np.array(ganancias)
        ganancias.view('i8,i8').sort(order=['f1'], axis=0)
        ganancias = ganancias[::-1]

        while nEval < maxEval:
            # Seleccionar P(t) torneo binario
            padres = []
            gananciasPadres = []
            for i in range(tamPoblacion*2):
                nEval += 1
                index = np.random.randint(tamPoblacion, size=2)
                cromo1, cromo2 = poblacion[index]
                ganancia1 = obtenerGanancia(cromo1, train[ejecucion][0], train[ejecucion][1])
                ganancia2 = obtenerGanancia(cromo2, train[ejecucion][0], train[ejecucion][1])

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
                hijo1 = cruceAritmetico(padres[aux], padres[aux+1])
                hijos.append(mutate(hijo1))
                gananciasHijos.append([p, obtenerGanancia(hijo1, train[ejecucion][0], train[ejecucion][1])])
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

        w = np.copy(poblacion[int(ganancias[0][0])])
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

        poblacion = np.random.uniform(0, 1, size=(30, datos.shape[1] - 1))

        ganancias = []
        for i in range(tamPoblacion):
            ganancias.append([i, obtenerGanancia(poblacion[i], train[ejecucion][0], train[ejecucion][1])])

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
                ganancia1 = obtenerGanancia(cromo1, train[ejecucion][0], train[ejecucion][1])
                ganancia2 = obtenerGanancia(cromo2, train[ejecucion][0], train[ejecucion][1])

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
            gananciasHijos.append([0, obtenerGanancia(hijo1, train[ejecucion][0], train[ejecucion][1])])
            gananciasHijos.append([1, obtenerGanancia(hijo2, train[ejecucion][0], train[ejecucion][1])])

            hijos = np.array(hijos)
            gananciasHijos = np.array(gananciasHijos)

            if gananciasHijos[0][1] >= gananciasHijos[1][1]:
                mejor = gananciasHijos[0]
                peor = gananciasHijos[1]
            else:
                mejor = gananciasHijos[1]
                peor = gananciasHijos[0]

            if mejor[1] > ganancias[1][1]:
                poblacion[ int(ganancias[1][0]) ] = hijos[ int(mejor[0]) ]
                ganancias[1] = mejor
                if peor[1] > ganancias[0][1]:
                    poblacion[ int(ganancias[0][0]) ] = hijos[ int(peor[0]) ]
                    ganancias[0] = peor
                elif mejor[1] > ganancias[0][1]:
                    poblacion[int(ganancias[0][0])] = hijos[int(mejor[0])]
                    ganancias[0] = peor

            ganancias.view('i8,i8').sort(order=['f1'], axis=0)


        w = np.copy(poblacion[ int(ganancias[ganancias.shape[0]-1][0]) ])
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

        poblacion = np.random.uniform(0, 1, size=(30, datos.shape[1] - 1))

        ganancias = []
        for i in range(tamPoblacion):
            ganancias.append([i, obtenerGanancia(poblacion[i], train[ejecucion][0], train[ejecucion][1])])

        ganancias = np.array(ganancias)
        ganancias.view('i8,i8').sort(order=['f1'], axis=0)

        while nEval < maxEval:
            # Seleccionar P(t) torneo binario
            padres = []
            gananciasPadres = []
            for i in range(4):
                nEval += 1
                index = np.random.randint(tamPoblacion, size=2)
                cromo1, cromo2 = poblacion[index]
                ganancia1 = obtenerGanancia(cromo1, train[ejecucion][0], train[ejecucion][1])
                ganancia2 = obtenerGanancia(cromo2, train[ejecucion][0], train[ejecucion][1])

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
            hijo1 = cruceAritmetico(padres[0], padres[1])
            hijo2 = cruceAritmetico(padres[2], padres[3])
            hijos.append(mutate(hijo1))
            hijos.append(mutate(hijo2))
            gananciasHijos.append([0, obtenerGanancia(hijo1, train[ejecucion][0], train[ejecucion][1])])
            gananciasHijos.append([1, obtenerGanancia(hijo2, train[ejecucion][0], train[ejecucion][1])])

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

        w = np.copy(poblacion[int(ganancias[ganancias.shape[0] - 1][0])])
        resultados.append(evaluar(w, test[ejecucion][0], test[ejecucion][1], antes))



# -----------------------------------------------------------------------------------
# ------------------------ Algoritmo Memético (10 - 1.0) --------------------------
# -----------------------------------------------------------------------------------
elif opcion == '7':
    maxEval = 15000
    tamPoblacion = 10

    for ejecucion in range(5):
        print(ejecucion)
        antes = time.time()
        nEval = 0
        nGeneraciones = 0

        poblacion = np.random.uniform(0, 1, size=(30, datos.shape[1] - 1))

        ganancias = []
        for i in range(tamPoblacion):
            ganancias.append([i, obtenerGanancia(poblacion[i], train[ejecucion][0], train[ejecucion][1])])

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
                ganancia1 = obtenerGanancia(cromo1, train[ejecucion][0], train[ejecucion][1])
                ganancia2 = obtenerGanancia(cromo2, train[ejecucion][0], train[ejecucion][1])

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
                gananciasHijos.append([p, obtenerGanancia(hijo1, train[ejecucion][0], train[ejecucion][1])])
                gananciasHijos.append([p + 1, obtenerGanancia(hijo2, train[ejecucion][0], train[ejecucion][1])])

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
                    gananciaAntigua = obtenerGanancia(hijos[i], train[ejecucion][0], train[ejecucion][1])

                    while nAtrib < (2 * datos.shape[1]) and fin == False:
                        hijosBL = np.copy(hijos)
                        nEval += 1

                        for posNuevo in np.random.permutation(datos.shape[1] - 1):
                            hijosBL[i][posNuevo] += np.random.normal(loc=0, scale=(0.3 ** 2))
                            hijosBL[i][posNuevo] = np.clip(hijos[i][posNuevo], 0, 1)
                            gananciaNueva = obtenerGanancia(hijosBL[i], train[ejecucion][0], train[ejecucion][1])

                            if gananciaNueva > gananciaAntigua:
                                gananciaAntigua = gananciaNueva
                                nAtrib = 0
                                break
                            else:
                                nAtrib += 1
                                hijos[i][posNuevo] = np.copy(hijosBL[i][posNuevo])

                            if nAtrib > 2 * datos.shape[1]:
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

        w = np.copy(poblacion[int(ganancias[0][0])])
        resultados.append(evaluar(w, test[ejecucion][0], test[ejecucion][1], antes))



# -----------------------------------------------------------------------------------
# ------------------------ Algoritmo Memético (10 - 0.1) --------------------------
# -----------------------------------------------------------------------------------
elif opcion == '8':
    maxEval = 15000
    tamPoblacion = 10

    for ejecucion in range(5):
        print(ejecucion)
        antes = time.time()
        nEval = 0
        nGeneraciones = 0

        poblacion = np.random.uniform(0, 1, size=(30, datos.shape[1] - 1))

        ganancias = []
        for i in range(tamPoblacion):
            ganancias.append([i, obtenerGanancia(poblacion[i], train[ejecucion][0], train[ejecucion][1])])

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
                ganancia1 = obtenerGanancia(cromo1, train[ejecucion][0], train[ejecucion][1])
                ganancia2 = obtenerGanancia(cromo2, train[ejecucion][0], train[ejecucion][1])

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
                gananciasHijos.append([p, obtenerGanancia(hijo1, train[ejecucion][0], train[ejecucion][1])])
                gananciasHijos.append([p + 1, obtenerGanancia(hijo2, train[ejecucion][0], train[ejecucion][1])])

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
                    gananciaAntigua = obtenerGanancia(hijos[i], train[ejecucion][0], train[ejecucion][1])

                    while nAtrib < (2 * datos.shape[1]) and fin == False:
                        hijosBL = np.copy(hijos)
                        nEval += 1

                        for posNuevo in np.random.permutation(datos.shape[1] - 1):
                            hijosBL[i][posNuevo] += np.random.normal(loc=0, scale=(0.3 ** 2))
                            hijosBL[i][posNuevo] = np.clip(hijos[i][posNuevo], 0, 1)
                            gananciaNueva = obtenerGanancia(hijosBL[i], train[ejecucion][0], train[ejecucion][1])

                            if gananciaNueva > gananciaAntigua:
                                gananciaAntigua = gananciaNueva
                                nAtrib = 0
                                break
                            else:
                                nAtrib += 1
                                hijos[i][posNuevo] = np.copy(hijosBL[i][posNuevo])

                            if nAtrib > 2 * datos.shape[1]:
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

        w = np.copy(poblacion[int(ganancias[0][0])])
        resultados.append(evaluar(w, test[ejecucion][0], test[ejecucion][1], antes))



# -----------------------------------------------------------------------------------
# ------------------------ Algoritmo Memético (10 - 0.1 mejores) --------------------------
# -----------------------------------------------------------------------------------
elif opcion == '9':
    maxEval = 15000
    tamPoblacion = 10

    for ejecucion in range(5):
        print(ejecucion)
        antes = time.time()
        nEval = 0
        nGeneraciones = 0

        poblacion = np.random.uniform(0, 1, size=(30, datos.shape[1] - 1))

        ganancias = []
        for i in range(tamPoblacion):
            ganancias.append([i, obtenerGanancia(poblacion[i], train[ejecucion][0], train[ejecucion][1])])

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
                ganancia1 = obtenerGanancia(cromo1, train[ejecucion][0], train[ejecucion][1])
                ganancia2 = obtenerGanancia(cromo2, train[ejecucion][0], train[ejecucion][1])

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
                gananciasHijos.append([p, obtenerGanancia(hijo1, train[ejecucion][0], train[ejecucion][1])])
                gananciasHijos.append([p + 1, obtenerGanancia(hijo2, train[ejecucion][0], train[ejecucion][1])])

            for p in range(int(tamPoblacion * 0.7), tamPoblacion):
                hijos.append(padres[p])
                gananciasHijos.append(gananciasPadres[p])

            hijos = np.array(hijos)
            gananciasHijos = np.array(gananciasHijos)
            gananciasHijos.view('i8,i8').sort(order=['f1'], axis=0)
            gananciasHijos = gananciasHijos[::-1]

            # Busqueda Local
            if nGeneraciones % 10 == 0:
                for i in range(int(tamPoblacion*0.1)):
                    nAtrib = 0
                    fin = False
                    gananciaAntigua = obtenerGanancia(hijos[i], train[ejecucion][0], train[ejecucion][1])

                    while nAtrib < (2 * datos.shape[1]) and fin == False:
                        hijosBL = np.copy(hijos)

                        for posNuevo in np.random.permutation(datos.shape[1] - 1):
                            nEval += 1
                            hijosBL[i][posNuevo] += np.random.normal(loc=0, scale=(0.3 ** 2))
                            hijosBL[i][posNuevo] = np.clip(hijos[i][posNuevo], 0, 1)
                            gananciaNueva = obtenerGanancia(hijosBL[i], train[ejecucion][0], train[ejecucion][1])

                            if gananciaNueva > gananciaAntigua:
                                gananciaAntigua = gananciaNueva
                                nAtrib = 0
                                break
                            else:
                                nAtrib += 1
                                hijos[i][posNuevo] = np.copy(hijosBL[i][posNuevo])

                            if nAtrib > 2 * datos.shape[1]:
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

        w = np.copy(poblacion[int(ganancias[0][0])])
        resultados.append(evaluar(w, test[ejecucion][0], test[ejecucion][1], antes))


else:
    print("Opción incorrecta")


df = pd.DataFrame(resultados)
df.to_csv('result.csv', index=False)
print(df)