# Student Score Predictor

A simple **Flask + Machine Learning (Linear Regression)** web app that predicts a student’s exam percentage based on study hours per day.  

This project was built as part of my **B.Sc. Data Science project** 🎓.  

Render Deployment Link -> https://student-score-predictor-3goj.onrender.com/
---

## Features
- Enter daily study hours → predict exam percentage 
- Clean & interactive web UI with modern CSS
- Auto trains on data after specific period of time(currently 10 minutes)
- Add your own data via Google Form 
- Deployable on **Heroku/Render** for easy sharing

---

## Usage
- Create Virutal Environment
    ```bash
    python -m venv venv
    ```
- Activate Virtual Environment
    ```bash
    #For linux
    source venv/bin/activate
    ```
    ```bash
    #For Windows
    .\venv\Scripts\Activate.ps1
    ```
    <mark>If you encounter issue during activating environment on windows, try executing this command first in powershell</mark>
    ```bash
    Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
    ```
    <mark>Then restart your terminal or close and open the Code editor again.</mark>
- Run the application
    ```bash
    python app.py
    ```

- Open your Browser and go to the url to access the application
    ```bash
    http://localhost:5000
    ```

# Message
Hi, I’m Sahil Gour (@SirSevrus) 👋

You’re welcome to use this project in your own work, and I’d love for you to collaborate if you’re interested. If you find this project helpful, that makes me really happy!

I’m planning to deploy it on render so that more people can access it easily. The project isn’t perfect yet, but I’m continuously working on improving it.