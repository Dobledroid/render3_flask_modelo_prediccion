from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('modelo_perezosos.pkl')
app.logger.debug('Modelo cargado correctamente.')

# Cargar el mapeo de clases original a codificadas
label_encoder = LabelEncoder()
label_encoder.classes_ = joblib.load('label_encoder_classes.pkl')
app.logger.debug('Mapeo de clases cargado correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        claw_length_cm = float(request.form['claw_length_cm'])
        size_cm = float(request.form['size_cm'])
        tail_length_cm = float(request.form['tail_length_cm'])
        
        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[size_cm, tail_length_cm, claw_length_cm]], 
                               columns=['size_cm', 'tail_length_cm', 'claw_length_cm'])
        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Realizar predicciones
        prediction_encoded = model.predict(data_df)
        app.logger.debug(f'Predicción (codificada): {prediction_encoded[0]}')
        
        # Invertir la codificación para obtener la especie original
        prediction_original = label_encoder.inverse_transform(prediction_encoded)
        app.logger.debug(f'Predicción (original): {prediction_original[0]}')
        
        # Devolver las predicciones como respuesta JSON
        return jsonify({'categoria': prediction_original[0]})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
