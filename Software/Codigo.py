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

for train_index, test_index in skf.split(X, y):
   x_train, y_train = X[train_index], y[train_index]
   x_test, y_test   = X[test_index], y[test_index]

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
print("10 - Enfriamiento Simulado (ES)")
print("11 - Iterative Local Search (ILS)")
print("12 - Evolución Diferencial (ED)")
opcion = input()

nEjecucuiones = 5
resultados = []


# -----------------------------------------------------------------------------------
# ---------------------------- Algoritmo Búsqueda Local -----------------------------
# -----------------------------------------------------------------------------------
if opcion == '1':
    for e in range(nEjecucuiones):
        antes = time.time()
        w = busquedaLocal(train[e], test[e], 15000, 20, X.shape[1])
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
        w = enfriamientoSimulado(train[e], 15000, 10, X.shape[1])
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
# -------------------------- Evolución Diferencial (DE).  ---------------------------
# -----------------------------------------------------------------------------------
elif opcion == '12':
    for e in range(nEjecucuiones):
        antes = time.time()
        w = evolucionDiferencial(train[e], 15000, 10, X.shape[1])
        resultados.append(evaluar(w, test[e][0], test[e][1], antes))



else:
    print("Opción incorrecta")


df = pd.DataFrame(resultados)
df.to_csv('result.csv', index=False)
print(df)