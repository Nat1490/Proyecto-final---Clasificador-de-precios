
# your code here
import os
from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import joblib
import json
# Inicializar la app Flask
app = Flask(__name__, template_folder=os.path.join(os.getcwd(), 'src/templates'))
# Cargar el modelo de XGBoost y los transformadores preentrenados
model = joblib.load('models/xgboost_model_optimized.pkl')
city_factorizer = joblib.load('models/city_factorizer.pkl')
state_factorizer = joblib.load('models/state_factorizer.pkl')
# Ruta absoluta al archivo JSON
json_path = 'data/interim/data_dict.json'
# Cargar el archivo JSON que contiene los estados y ciudades
try:
    with open(json_path, 'r') as f:
        data_dict = json.load(f)
        # Ordenar los estados y las ciudades alfabéticamente
        data_dict = {estado: sorted(ciudades) for estado, ciudades in sorted(data_dict.items())}
except FileNotFoundError:
    print(f"File not found: {json_path}")
    data_dict = {}
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
    data_dict = {}
except Exception as e:
    print(f"General error loading JSON: {e}")
    data_dict = {}
# Página principal: Mostrar el formulario de ingreso de datos
@app.route('/')
def index():
    if not data_dict:
        return "Error: data_dict not loaded correctly"
    estados = sorted(data_dict.keys())
    return render_template('index.html', estados=estados)
# Ruta para obtener las ciudades de un estado dado
@app.route('/get_cities', methods=['POST'])
def get_cities():
    state = request.json.get('state')
    cities = data_dict.get(state, [])
    return jsonify(cities)
# Ruta para manejar las predicciones
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos del formulario
        city = request.form['city']
        state = request.form['state']
        # Validar campos numéricos y asignar valores por defecto si están vacíos
        bed = float(request.form['bed']) if request.form['bed'] else 0
        bath = float(request.form['bath']) if request.form['bath'] else 0
        house_size = float(request.form['house_size']) if request.form['house_size'] else 0
        # Calcular el logaritmo del tamaño de la casa
        log_house_size = np.log(house_size) if house_size > 0 else 0
        # Manejar el jardín
        incluir_jardin = request.form.get('incluir_jardin') == 'on'
        if incluir_jardin:
            garden_percentage = float(request.form['garden_percentage']) if request.form['garden_percentage'] else 0
            garden_size = (garden_percentage / 100) * house_size
            acre_lot_sqft = house_size + garden_size
        else:
            acre_lot_sqft = house_size
        acre_lot = acre_lot_sqft / 43560
        # Capturar estado civil y número de hijos
        tiene_hijos = request.form['tiene_hijos']
        num_hijos = int(request.form['num_hijos']) if request.form.get('num_hijos') else 0
        estado_civil = request.form['estado_civil']
        # Variables salariales
        one_adult_no_kids_living_wage = 0
        one_adult_one_kid_living_wage = 0
        one_adult_two_kids_living_wage = 0
        one_adult_three_kids_living_wage = 0
        two_adults_one_working_no_kids_living_wage = 0
        two_adults_one_working_one_kid_living_wage = 0
        two_adults_one_working_two_kids_living_wage = 0
        two_adults_one_working_three_kids_living_wage = 0
        two_adults_both_working_no_kids_living_wage = 0
        two_adults_both_working_one_kid_living_wage = 0
        two_adults_both_working_two_kids_living_wage = 0
        two_adults_both_working_three_kids_living_wage = 0
        # Definir salarios según estado civil
        if estado_civil == 'soltero':
            salario = float(request.form['salario_soltero']) if request.form.get('salario_soltero') else 0
            if tiene_hijos == 'no':
                one_adult_no_kids_living_wage = salario
            elif num_hijos == 1:
                one_adult_one_kid_living_wage = salario
            elif num_hijos == 2:
                one_adult_two_kids_living_wage = salario
            elif num_hijos == 3:
                one_adult_three_kids_living_wage = salario
        elif estado_civil == 'pareja':
            salario1 = float(request.form['salario_pareja1']) if request.form.get('salario_pareja1') else 0
            salario2 = float(request.form['salario_pareja2']) if request.form.get('salario_pareja2') else 0
            trabajadores = request.form['trabajadores']
            if trabajadores == 'uno':
                if tiene_hijos == 'no':
                    two_adults_one_working_no_kids_living_wage = salario1
                elif num_hijos == 1:
                    two_adults_one_working_one_kid_living_wage = salario1
                elif num_hijos == 2:
                    two_adults_one_working_two_kids_living_wage = salario1
                elif num_hijos == 3:
                    two_adults_one_working_three_kids_living_wage = salario1
            else:
                combined_salary = salario1 + salario2
                if tiene_hijos == 'no':
                    two_adults_both_working_no_kids_living_wage = combined_salary
                elif num_hijos == 1:
                    two_adults_both_working_one_kid_living_wage = combined_salary
                elif num_hijos == 2:
                    two_adults_both_working_two_kids_living_wage = combined_salary
                elif num_hijos == 3:
                    two_adults_both_working_three_kids_living_wage = combined_salary
        # Factorizar 'city' y 'state' usando los transformadores preentrenados
        city_factorized = np.where(city_factorizer == city)[0][0] if city in city_factorizer else -1
        state_factorized = np.where(state_factorizer == state)[0][0] if state in state_factorizer else -1
        if city_factorized == -1 or state_factorized == -1:
            return "Error: Ciudad o estado no reconocidos."
        # Preparar los datos de entrada en formato DataFrame
        input_data = pd.DataFrame({
            'bed': [bed],
            'bath': [bath],
            'acre_lot': [acre_lot],
            'log_house_size': [log_house_size],
            'one_adult_no_kids_living_wage': [one_adult_no_kids_living_wage],
            'one_adult_one_kid_living_wage': [one_adult_one_kid_living_wage],
            'one_adult_two_kids_living_wage': [one_adult_two_kids_living_wage],
            'one_adult_three_kids_living_wage': [one_adult_three_kids_living_wage],
            'two_adults_one_working_no_kids_living_wage': [two_adults_one_working_no_kids_living_wage],
            'two_adults_one_working_one_kid_living_wage': [two_adults_one_working_one_kid_living_wage],
            'two_adults_one_working_two_kids_living_wage': [two_adults_one_working_two_kids_living_wage],
            'two_adults_one_working_three_kids_living_wage': [two_adults_one_working_three_kids_living_wage],
            'two_adults_both_working_no_kids_living_wage': [two_adults_both_working_no_kids_living_wage],
            'two_adults_both_working_one_kid_living_wage': [two_adults_both_working_one_kid_living_wage],
            'two_adults_both_working_two_kids_living_wage': [two_adults_both_working_two_kids_living_wage],
            'two_adults_both_working_three_kids_living_wage': [two_adults_both_working_three_kids_living_wage],
            'crime_index': [0],  # Asigna el valor real de 'crime_index' si está disponible
            'city_n': [city_factorized],
            'state_n': [state_factorized]
        })
        # Hacer la predicción
        prediction_log = model.predict(input_data)
        prediction = np.exp(prediction_log)  # Convertir de logaritmo a valor real
        # Enviar el resultado al cliente
        return render_template('result.html', prediction=prediction[0])
    except Exception as e:
        return f"Error en la predicción: {str(e)}"
if __name__ == '__main__':
    app.run(debug=True)