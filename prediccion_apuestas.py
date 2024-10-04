import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def cargar_datos():
    # Simulamos un dataset pequeño (goles_local, goles_visitante, posesion_local, posesion_visitante, resultado)
    data = {
        'goles_local': [2, 1, 3, 1, 0, 2, 1, 0, 3, 2],
        'goles_visitante': [1, 0, 1, 2, 1, 1, 0, 1, 2, 3],
        'posesion_local': [55, 48, 60, 52, 50, 57, 49, 51, 60, 54],
        'posesion_visitante': [45, 52, 40, 48, 50, 43, 51, 49, 40, 46],
        'resultado': [1, 1, 1, 0, 0, 1, 1, 0, 0, 1]  # 1 = Victoria local, 0 = Empate o derrota local
    }
    return pd.DataFrame(data)


def entrenar_modelo():
    data = cargar_datos()
    X = data[['goles_local', 'goles_visitante', 'posesion_local', 'posesion_visitante']]
    y = data['resultado']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)

    
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Precisión del modelo: {accuracy * 100:.2f}%')

    return modelo


def pedir_datos_partido():
    print("Introduce los datos del partido para hacer una predicción:")
    goles_local = int(input("Goles del equipo local: "))
    goles_visitante = int(input("Goles del equipo visitante: "))
    posesion_local = float(input("Porcentaje de posesión del equipo local: "))
    posesion_visitante = float(input("Porcentaje de posesión del equipo visitante: "))
    
    return [[goles_local, goles_visitante, posesion_local, posesion_visitante]]


def predecir_partido(modelo):
    nuevo_partido = pedir_datos_partido()
    resultado = modelo.predict(nuevo_partido)
    
    if resultado[0] == 1:
        print("Predicción: El equipo local ganará.")
    else:
        print("Predicción: El equipo local no ganará (empate o derrota).")

# Main
if __name__ == "__main__":
    modelo = entrenar_modelo()
    predecir_partido(modelo)
