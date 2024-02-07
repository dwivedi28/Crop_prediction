from flask import Flask, request, render_template

from src.pipelines.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application


@app.route('/')
def home():
    return render_template('Home_1.html')


@app.route('/Predict')
def prediction():
    return render_template('Index.html')


@app.route('/form', methods=["POST"])
def brain():
    try:
        data = CustomData(
            N=float(request.form['N']),
            P=float(request.form['P']),
            K=float(request.form['K']),
            temperature=float(request.form['temperature']),
            humidity=float(request.form['humidity']),
            ph=float(request.form['ph']),
            rainfall=float(request.form['rainfall'])
        )

        if 0 < data.ph <= 14 and 0 < data.temperature < 100 and data.humidity > 0:
            final_new_data = data.get_data_as_dataframe()
            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(final_new_data)
            results = str(pred)

            return render_template('prediction.html', final_result=results)
        else:
            return "Sorry...  Error in entered values in the form. Please check the values and fill it again"
    except Exception as e:
        return f"Error occurred: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True)
