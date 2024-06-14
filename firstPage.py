import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import joblib
from sklearn.impute import SimpleImputer


app = Flask(__name__)

# Load the training dataset and perform basic data preprocessing
filename = app.static_folder + '/dataset/cleaned.csv'
data = pd.read_csv(filename)
data_columns = ["ID", "Price", "Levy", "Manufacturer", "Model", "Prod. year", "Category", "Leather interior",
                "Fuel type", "Engine volume", "Mileage", "Cylinders", "Gear box type", "Drive wheels", "Wheel", "Color", "Airbags"]

# Extract features and target variable
features = data.drop('Price', axis=1)

# Select categorical columns
cat_columns = features.select_dtypes(include='O').columns

# Convert categorical columns to dummy/indicator variables
features = pd.get_dummies(features, columns=cat_columns, drop_first=True)

# Load the Extra Trees model
extra_trees_model = joblib.load('C:\\Users\\Zakaria\\Documents\\car price prediction project\\extra_trees_regressor_model.pkl')

# Load the Decision Tree model
decision_tree_model = joblib.load(r'C:\Users\Zakaria\Documents\car price prediction project\decision_trees_regressor.pkl')

# Load the RandomForestRegressor model
forest_model = joblib.load(r'C:\Users\Zakaria\Documents\car price prediction project\forest_model.pkl')

# Load the GradientBoostingRegressor model
gradient_model = joblib.load(r'C:\Users\Zakaria\Documents\car price prediction project\gradient_model.pkl')


# Create an imputer with a strategy (e.g., mean)
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on your training data
imputer.fit(features)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/data')
def data():
    filename = app.static_folder + '/dataset/cleaned.csv'
    data = pd.read_csv(filename)
    data_columns = ["Price", "Levy", "Manufacturer", "Model", "Prod. year", "Category", "Leather interior", "Fuel type", "Engine volume", "Mileage", "Cylinders", "Gear box type", "Drive wheels","Wheel", "Color", "Airbags"]

    return render_template('data.html', data=data, data_columns=data_columns)


@app.route('/enter_parameters', methods=['GET', 'POST'])
def enter_parameters():
    if request.method == 'POST':
        # Extract form data
        form_data = {
            'Levy': float(request.form['levy']),
            'Manufacturer': request.form['manufacturer'],
            'Model': request.form['model'],
            'Prod. year': int(request.form['prodYear']),
            'Category': request.form['category'],
            'Fuel type': request.form['fuel_type'],
            'Engine volume': float(request.form['engineVolume']),
            'Mileage': int(request.form['mileage']),
            'Cylinders': float(request.form['cylinders']),
            'Gear box type': request.form['gear_box_type'],
            'Wheel': request.form['wheel'],
            'Color': request.form['color'],
            'Drive wheels': request.form['driveWheels'],
            'Leather interior': request.form.get('leatherInterior'),
            'Airbags': int(request.form['mileage'])
        }
        
        observation = pd.DataFrame([form_data], columns=features.columns)
        observation_imputed = pd.DataFrame(imputer.transform(observation), columns=observation.columns)

        decision_tree_prediction = decision_tree_model.predict(observation)
        rounded_decision_tree_prediction = np.round(decision_tree_prediction[0], 2)

        extra_trees_prediction = extra_trees_model.predict(observation)
        rounded_extra_trees_prediction = np.round(extra_trees_prediction[0], 2)

        forest_prediction = forest_model.predict(observation_imputed)
        rounded_forest_prediction = np.round(forest_prediction[0], 2)

        gradient_prediction = gradient_model.predict(observation_imputed)
        rounded_gradient_prediction = np.round(gradient_prediction[0], 2)

        return redirect(url_for('models',
                                decision_tree_prediction=rounded_decision_tree_prediction,
                                extra_trees_prediction=rounded_extra_trees_prediction,
                                forest_prediction=rounded_forest_prediction,
                                gradient_prediction=rounded_gradient_prediction))

    # Rest of the code remains the same
    filename = app.static_folder + '/dataset/cleaned.csv'
    data = pd.read_csv(filename)
    data_columns = ["ID", "Price", "Levy", "Manufacturer", "Model", "Prod. year", "Category", "Leather interior", "Fuel type", "Engine volume", "Mileage", "Cylinders", "Gear box type", "Drive wheels", "Wheel", "Color", "Airbags"]

    unique_manufacturers = data["Manufacturer"].unique()
    unique_categories = data['Category'].unique()
    unique_fuel_types = data['Fuel type'].unique()
    unique_gear_box_types = data['Gear box type'].unique()
    unique_wheel = data['Wheel'].unique()

    return render_template('enter_parameters.html', unique_wheel=unique_wheel,  unique_gear_box_types=unique_gear_box_types, unique_fuel_types=unique_fuel_types, unique_manufacturers=unique_manufacturers, unique_categories=unique_categories, data_columns=data_columns)


@app.route('/models/<float:decision_tree_prediction>/<float:extra_trees_prediction>/<float:forest_prediction>/<float:gradient_prediction>')
def models(decision_tree_prediction, extra_trees_prediction, forest_prediction, gradient_prediction):
    return render_template('models.html',
                           decision_tree_prediction=decision_tree_prediction,
                           extra_trees_prediction=extra_trees_prediction,
                           forest_prediction=forest_prediction,
                           gradient_prediction=gradient_prediction)

@app.route('/models/decision_tree_regressor/<float:decision_tree_prediction>')
def decision_tree_regressor(decision_tree_prediction):
    return render_template('decision_tree_regressor.html', decision_tree_prediction=decision_tree_prediction)



@app.route('/models/gradient/<float:gradient_prediction>')
def gradient(gradient_prediction):
    return render_template('gradient.html',gradient_prediction=gradient_prediction)

@app.route('/models/random_forest_model/<float:forest_prediction>')
def random_forest_model(forest_prediction):
    return render_template('random_forest_model.html', forest_prediction=forest_prediction)

@app.route('/models/extra/<float:extra_trees_prediction>')
def extra(extra_trees_prediction):
    return render_template('extra.html',extra_trees_prediction=extra_trees_prediction)

if __name__ == '__main__':
    app.run(debug=True)

