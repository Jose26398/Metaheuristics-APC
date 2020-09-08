from previousPractices import *

np.random.seed(1)


# Lectura de datos
datos = read_data('doc/texture.csv')

#División de los datos
X, y, train, test = divide_data(datos)



# -----------------------------------------------------------------------------------
# Menu de control de la aplicación
# (Sin relación con la práctica, solo para facilizar su uso)
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
print("10 - Enfriamiento Simulado (ES)")
print("11 - Iterative Local Search (ILS)")
print("12 - Evolución Diferencial Aleatorio (ED-A)")
print("13 - Evolución Diferencial Mejor+Aleatorio (ED-M)")
opcion = input()

nEjecucuiones = 5
resultados = []


# -----------------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------------
def enfriamientoSimulado(train, maxEval, numAtributos):
    w = np.random.random_sample((numAtributos,))
    ganancia = obtenerGanancia(w, train[0], train[1])
    mejorW = np.copy(w)
    mejorGanancia = np.copy(ganancia)

    maxAtrib = 10 * numAtributos
    M = maxEval / maxAtrib
    maxExitos = 0.1 * maxAtrib

    temperatura = (0.3*ganancia)/-(np.log(0.3))
    if (temperatura < 0.001):
        temperatura = 0.001

    contEval = 0
    exitos = 1
    while contEval < maxEval and exitos != 0:
        contAtrib = 0
        exitos = 0

        while contAtrib < maxAtrib and exitos < maxExitos:
            wAux = mutate(w)
            gananciaAux = obtenerGanancia(w, train[0], train[1])
            contEval += 1

            if (gananciaAux - ganancia) > 0 or np.random.rand() >=\
                np.exp((gananciaAux - ganancia)/temperatura):
                w = np.copy(wAux)
                ganancia = np.copy(gananciaAux)
                exitos += 1
                if mejorGanancia < ganancia:
                    mejorW = np.copy(w)
                    ganancia = np.copy(gananciaAux)

            contAtrib += 1

        beta = (temperatura - 0.001) / M*temperatura*0.001
        temperatura = temperatura / (1 + (beta*temperatura))
        if (temperatura < 0.001):
            temperatura = 0.001

    return mejorW


# -----------------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------------
def mutateILS(w, t):
    for i in t:
        w[i] += np.random.normal(loc=0, scale=(0.4**2))
        w[i] = np.clip(w[i], 0, 1)

        return w


# -----------------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------------
def iterativeLS(train, maxEval, maxVec, numAtributos):
    w = np.random.random_sample((numAtributos,))
    w = busquedaLocal(w, train, 1000, maxVec, numAtributos)
    ganancia = obtenerGanancia(w, train[0], train[1])

    i = 1000
    while i < maxEval:
        i += 1000
        t = random.sample(list(range(0, numAtributos)), int(0.1 * numAtributos))
        wAux = mutateILS(w, t)
        wAux = busquedaLocal(wAux, train, 1000, maxVec, numAtributos)
        gananciaAux = obtenerGanancia(wAux, train[0], train[1])

        if gananciaAux > ganancia:
            w = np.copy(wAux)
            ganancia = np.copy(gananciaAux)

    return w


# -----------------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------------
def cruceAleatorio(p1, p2, p3, F):
    cruce = (p1 + F*(p2-p3))
    return np.clip(cruce, 0, 1)


# -----------------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------------
def cruceMejor(w, best, p1, p2, F):
    cruce = (w + F*(best-w) + F*(p1-p2))
    return np.clip(cruce, 0, 1)


# -----------------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------------
def evolucionDiferencial(train, maxEval, numAtributos, CR, tipoCruce):
    tamPoblacion = 50
    poblacion = np.random.uniform(0, 1, size=(tamPoblacion, numAtributos))

    ganancias = []
    for i in range(tamPoblacion):
        ganancias.append([obtenerGanancia(poblacion[i], train[0], train[1])])

    ganancias = np.array(ganancias)
    nEval = poblacion.shape[0]

    while nEval < maxEval:

        for i in range(tamPoblacion):
            index = random.sample(list(range(0, tamPoblacion)), 3)
            padre1, padre2, padre3 = poblacion[index]

            posW = np.random.randint(0, tamPoblacion)
            w = np.copy(poblacion[posW])
            ganancia = np.copy(ganancias[posW])

            wAux = np.copy(w)
            for j in range(numAtributos):
                if np.random.rand() <= CR and tipoCruce == 'Aleatorio':
                    wAux[j] = cruceAleatorio(padre1[j], padre2[j], padre3[j], 0.5)
                if np.random.rand() <= CR and tipoCruce == 'Mejor':
                    mejor = poblacion[ np.where(max(ganancias)) ].reshape(numAtributos)
                    wAux[j] = cruceMejor(w[j], mejor[j], padre1[j], padre2[j], 0.5)

            gananciaAux = obtenerGanancia(poblacion[i], train[0], train[1])
            nEval += 1

            if gananciaAux < ganancia:
                poblacion[posW] = np.copy(wAux)
                ganancias[posW] = np.copy(gananciaAux)


    return poblacion[ np.where(max(ganancias)) ].reshape(numAtributos)




# -----------------------------------------------------------------------------------
# ---------------------------- Algoritmo Búsqueda Local -----------------------------
# -----------------------------------------------------------------------------------
if opcion == '1':
    for e in range(nEjecucuiones):
        antes = time.time()
        w = np.random.random_sample((X.shape[1],))
        w = busquedaLocal(w, train[e], 15000, 20, X.shape[1])
        resultados.append(evaluar(w, test[e][0], test[e][1], antes))



# -----------------------------------------------------------------------------------
# ----------------------------- Algoritmo Gredy RELIEF ------------------------------
# -----------------------------------------------------------------------------------
elif opcion == '2':

    for e in range(nEjecucuiones):
        antes = time.time()
        w = greedyRELIEF(train[e], X.shape[1])
        resultados.append(evaluar(w, test[e][0], test[e][1], antes))




# -----------------------------------------------------------------------------------
# ------------------ Algoritmo Genético Generacional (Cruce BLX) --------------------
# -----------------------------------------------------------------------------------
elif opcion == '3':
    for e in range(nEjecucuiones):
        antes = time.time()
        w = geneticoGeneracionalBLX(train[e], 15000, 30, X.shape[1])
        resultados.append(evaluar(w, test[e][0], test[e][1], antes))



# ------------------------------------------------------------------------------------
# --------------- Algoritmo Genético Generacional (Cruce Aritmético) -----------------
# ------------------------------------------------------------------------------------
elif opcion == '4':
    for e in range(nEjecucuiones):
        antes = time.time()
        w = geneticoGeneracionalA(train[e], 15000, 30, X.shape[1])
        resultados.append(evaluar(w, test[e][0], test[e][1], antes))




# -----------------------------------------------------------------------------------
# ------------------ Algoritmo Genético Estacionario (Cruce BLX) --------------------
# -----------------------------------------------------------------------------------
elif opcion == '5':
    for e in range(nEjecucuiones):
        antes = time.time()
        w = geneticoEstacionarioBLX(train[e], 15000, 30, X.shape[1])
        resultados.append(evaluar(w, test[e][0], test[e][1], antes))




# ------------------------------------------------------------------------------------
# --------------- Algoritmo Genético Estacionario (Cruce Aritmético) -----------------
# ------------------------------------------------------------------------------------
elif opcion == '6':
    for e in range(nEjecucuiones):
        antes = time.time()
        w = geneticoEstacionarioA(train[e], 15000, 30, X.shape[1])
        resultados.append(evaluar(w, test[e][0], test[e][1], antes))



# -----------------------------------------------------------------------------------
# ------------------------- Algoritmo Memético (10 - 1.0) ---------------------------
# -----------------------------------------------------------------------------------
elif opcion == '7':
    for e in range(nEjecucuiones):
        antes = time.time()
        w = memetico10_10(train[e], 15000, 10, X.shape[1])
        resultados.append(evaluar(w, test[e][0], test[e][1], antes))



# -----------------------------------------------------------------------------------
# ------------------------- Algoritmo Memético (10 - 0.1) ---------------------------
# -----------------------------------------------------------------------------------
elif opcion == '8':
    for e in range(nEjecucuiones):
        antes = time.time()
        w = memetico10_01(train[e], 15000, 10, X.shape[1])
        resultados.append(evaluar(w, test[e][0], test[e][1], antes))



# -----------------------------------------------------------------------------------
# --------------------- Algoritmo Memético (10 - 0.1 mejores) -----------------------
# -----------------------------------------------------------------------------------
elif opcion == '9':
    for e in range(nEjecucuiones):
        antes = time.time()
        w = memetico10_01_mej(train[e], 15000, 10, X.shape[1])
        resultados.append(evaluar(w, test[e][0], test[e][1], antes))



# -----------------------------------------------------------------------------------
# --------------------------- Enfriamiento Simulado (ES) ----------------------------
# -----------------------------------------------------------------------------------
elif opcion == '10':
    for e in range(nEjecucuiones):
        antes = time.time()
        w = enfriamientoSimulado(train[e], 15000, X.shape[1])
        resultados.append(evaluar(w, test[e][0], test[e][1], antes))



# -----------------------------------------------------------------------------------
# ------------------------ Búsqueda Local Reiterada (ILS)  --------------------------
# -----------------------------------------------------------------------------------
elif opcion == '11':
    for e in range(nEjecucuiones):
        antes = time.time()
        w = iterativeLS(train[e], 15000, 10, X.shape[1])
        resultados.append(evaluar(w, test[e][0], test[e][1], antes))



# -----------------------------------------------------------------------------------
# ----------------- Evolución Diferencial (DE) - Cruce Aleatorio  -------------------
# -----------------------------------------------------------------------------------
elif opcion == '12':
    for e in range(nEjecucuiones):
        antes = time.time()
        w = evolucionDiferencial(train[e], 15000, X.shape[1], 0.5, 'Aleatorio')
        resultados.append(evaluar(w, test[e][0], test[e][1], antes))



# -----------------------------------------------------------------------------------
# -------------- Evolución Diferencial (DE) - Cruce Mejor+Aleatorio  ----------------
# -----------------------------------------------------------------------------------
elif opcion == '13':
    for e in range(nEjecucuiones):
        antes = time.time()
        w = evolucionDiferencial(train[e], 15000, X.shape[1], 0.5, 'Mejor')
        resultados.append(evaluar(w, test[e][0], test[e][1], antes))


else:
    print("Opción incorrecta")


df = pd.DataFrame(resultados)
df.to_csv('result.csv', index=True)
print(df)