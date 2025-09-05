"""
Author: Sahil Gour (@SirSevrus)
"""

import os
import re
import io
import base64
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for Flask
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression

# preparing data and filtering it
def create_dataframe():
    df = pd.read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vTq08vBQuHiahx195CsqRy_kCyRwAQ6eqSa9L1kaZn5gXpSlNAOgGAGnIVe2z4HP8Uu2djZIcADEroC/pub?gid=1413079117&single=true&output=csv")
    df = df[['How many hours did you study per day (on average)? ',
             'Your exam percentage (out of 100)?']].copy()
    df.columns = ['Hours', 'Score']

    def clean_hours(val):
        if pd.isna(val):
            return None
        val = str(val).lower().strip()
        val = re.sub(r'(hours?|hr)', '', val)
        if "to" in val:
            parts = re.findall(r"[\d\.]+", val)
            if len(parts) == 2:
                return (float(parts[0]) + float(parts[1])) / 2
        match = re.search(r"[\d\.]+", val)
        return float(match.group()) if match else None

    def clean_score(val):
        if pd.isna(val):
            return None
        val = str(val).strip().replace("%", "")
        try:
            return float(val)
        except:
            return None

    df['Hours'] = df['Hours'].apply(clean_hours)
    df['Score'] = df['Score'].apply(clean_score)
    df = df.dropna(subset=['Hours', 'Score']).reset_index(drop=True)
    return df

# Training model
def train_model(df):
    X = df[['Hours']]
    y = df[['Score']]
    model = LinearRegression()
    model.fit(X, y)
    return model

# Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    df = create_dataframe()
    model = train_model(df)

    prediction = None
    plot_url = None

    if request.method == "POST":
        if "hours" in request.form:  # Normal prediction mode
            hours = float(request.form["hours"])
            input_df = pd.DataFrame([[hours]], columns=["Hours"])
            pred = model.predict(input_df)[0][0]
            prediction = max(0, min(100, pred))

        if "debug" in request.form:  # Debug mode → Generate scatter plot
            plt.figure(figsize=(6, 4))
            plt.scatter(df["Hours"], df["Score"], color="blue", label="Data Points")

            # Regression line
            X_line = pd.DataFrame(sorted(df["Hours"]), columns=["Hours"])
            y_line = model.predict(X_line)
            plt.plot(X_line, y_line, color="red", label="Regression Line")

            plt.xlabel("Study Hours per Day")
            plt.ylabel("Exam Score (%)")
            plt.title("Study Hours vs Exam Score")
            plt.legend()

            # Save to base64 for HTML display
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plot_url = base64.b64encode(buf.getvalue()).decode("utf-8")
            buf.close()
            plt.close()

    return render_template("index.html", prediction=prediction, plot_url=plot_url)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
