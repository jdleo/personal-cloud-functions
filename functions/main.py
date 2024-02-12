"""This module is used as an entry point for the application.

This will house various cloud functions. 
"""

from textwrap import dedent
from firebase_admin import initialize_app
from firebase_functions import https_fn, options
from sklearn.linear_model import LinearRegression
import flask
import pandas as pd
from flask import request

initialize_app()
app = flask.Flask(__name__)


@app.get("/health")
def ping():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict_weight")
def predict_weight():
    """Use Linear Regression to help predict weight loss from Apple Health samples."""

    # Get the data from the request (which is a string blob inside "data" key)
    data = flask.request.json.get("data")

    # Converting data into a DataFrame
    data = [line.split(",") for line in data.strip().split("\n")]
    df = pd.DataFrame(data, columns=["date", "weight"])
    df["date"] = pd.to_datetime(df["date"])
    df["weight"] = df["weight"].astype(float)

    # Average/smooth out missing data
    for i in range(len(df) - 1):
        current_date = df["date"].iloc[i]
        next_date = df["date"].iloc[i + 1]
        if (next_date - current_date).days > 1:
            for j in range(1, (next_date - current_date).days):
                new_date = current_date + pd.Timedelta(days=j)
                new_weight = (df["weight"].iloc[i] + df["weight"].iloc[i + 1]) / 2
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            [[new_date, new_weight]], columns=["date", "weight"]
                        ),
                    ]
                )

    # Converting dates to ordinal for linear regression
    df["date_ordinal"] = pd.to_datetime(df["date"]).apply(lambda date: date.toordinal())

    # Preparing the linear regression model
    model = LinearRegression()
    x = df[["date_ordinal"]]
    y = df["weight"]
    model.fit(x, y)

    # Predicting values
    df["predicted_weight"] = model.predict(x)

    # Displaying the linear regression formula
    slope = model.coef_[0]
    intercept = model.intercept_
    linreg_formula = f"Weight = {slope:.2f} * DateOrdinal + {intercept:.2f}"

    # Predict the weight for 15, 30, 45, 60, 90 days from last date
    last_date = df["date"].iloc[-1]
    last_date_ordinal = last_date.toordinal()
    predictions = []
    for days in [15, 30, 45, 60, 90]:
        new_date_ordinal = last_date_ordinal + days
        new_date = pd.Timestamp.fromordinal(new_date_ordinal)
        new_weight = model.predict([[new_date_ordinal]])[0]
        predictions.append((new_date.strftime("%m/%d"), new_weight))

    # Formatting the response
    return dedent(
        f"""
        Linear Regression Formula:
        {linreg_formula}

        Predictions:
        - {predictions[0][0]}: {predictions[0][1]:.2f} lbs
        - {predictions[1][0]}: {predictions[1][1]:.2f} lbs
        - {predictions[2][0]}: {predictions[2][1]:.2f} lbs
        - {predictions[3][0]}: {predictions[3][1]:.2f} lbs
        - {predictions[4][0]}: {predictions[4][1]:.2f} lbs
    """
    )


@https_fn.on_request(memory=options.MemoryOption.GB_1)
def https_function(req: https_fn.Request):
    """Main entry point for the application."""
    with app.request_context(req.environ):
        return app.full_dispatch_request()
