import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

# app instance
app = Flask(__name__)

# loading dataset and model
df = pd.read_excel('./Course_price_prediction.xlsx')
rfr_model = pickle.load(open('./rf_reg_model.pkl', 'rb'))

# index page


@app.route('/', methods=['GET', 'POST'])
def index():
    Institute_brand_value = sorted(df['Institute_brand_value'].unique())
    Course = sorted(df['Course'].unique())
    Course_market = sorted(df['Course_market'].unique())
    Online_Live_Class = sorted(df['Online_Live_Class'].unique())
    # Offiline_Classes = sorted(df['Offiline_Classes'].unique())
    Location = sorted(df['Location'].unique())
    Course_Level = sorted(df['Course_Level'].unique())
    Infrastructure_cost = sorted(df['Infrastructure_cost'].unique())
    Competition_level = sorted(df['Competition_level'].unique())
    Certification = sorted(df['Certification'].unique())
    Placement = sorted(df['Placement'].unique())


    return render_template('index.html', brand_value=Institute_brand_value, Course=Course, Course_market=Course_market, online=Online_Live_Class, loc=Location, level=Course_Level, cost=Infrastructure_cost, comp=Competition_level, cert=Certification, plct=Placement)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    brand_value = request.form.get('brand_value')
    Course = request.form.get('Course')
    Course_market = request.form.get('Course_market')
    online_class = request.form.get('online')
    Location = request.form.get('loc')
    course_level = request.form.get('level')
    infra_cost = request.form.get('cost')
    comp_level = request.form.get('comp')
    certification = request.form.get('cert')
    course_duration = request.form.get('duration')
    study_material = request.form.get('material_cost')
    off_rent = request.form.get('rent')
    off_elec = request.form.get('electricity')
    miss_exp = request.form.get('expenses')
    acqu_cost = request.form.get('acquisition')
    placement = request.form.get('plct')


    data = {
        'Institute_brand_value': brand_value,
        'Course': Course,
        'Course_market': Course_market,
        'Online_Live_Class': online_class,
        'Location': Location,
        'Course_Level': course_level,
        'Course_Duration_Hours': course_duration,
        'Study_material_cost': study_material,
        'Office_rent': off_rent,
        'Infrastructure_cost': infra_cost,
        'Office_electricity_charges': off_elec,
        'Miscellaneous_expenses': miss_exp,
        'Cost_of_acquisition': acqu_cost,
        'Competition_level': comp_level,
        'Certification': certification,
        'Placement': placement
        }

    features = pd.DataFrame(data, index=[0])
    pred = rfr_model.predict(features)

    return render_template("predict.html", prediction=np.round(pred[0], 2))


if __name__ == "__main__":
    app.run(debug=True)
