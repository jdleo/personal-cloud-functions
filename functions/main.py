"""This module is used as an entry point for the application.

This will house various cloud functions. 
"""

from textwrap import dedent
from datetime import datetime, timedelta
from firebase_admin import initialize_app
from firebase_functions import https_fn, options
from sklearn.linear_model import LinearRegression
import flask
import pandas as pd

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


@app.post("/predict_money")
def predict_money():
    """Use iterative modeling to help predict financial health for the year"""

    # Get input values from the request
    data = flask.request.json
    starting_balance = data.get("starting_balance")
    daily_pay = data.get("daily_pay")
    bill_days = data.get("bill_days").split(",")
    bill_amounts = list(map(float, data.get("bill_amounts").split(",")))
    bill_expiry_dates = data.get("bill_expiry_dates").split(",")
    cashflow_dates = data.get("cashflow_dates").split(",")
    cashflow_amounts = list(map(float, data.get("cashflow_amounts").split(",")))

    # Get the starting and ending date of the simulation (end is a year ahead)
    start_date = datetime.now()
    end_date = start_date + timedelta(days=365)

    # Initialize variables
    balance = starting_balance
    current_date = start_date
    dates = []
    balances = []

    # Loop from today's date to end_date
    while current_date <= end_date:
        # Apply daily pay on weekdays
        if current_date.weekday() < 5:
            # Edge case, don't apply daily pay if this is ran past 11pm UTC TODAY
            if current_date.hour >= 23 and current_date.date() == datetime.now().date():
                pass
            else:
                balance += daily_pay

        # Apply bills
        for day, amount, expiry in zip(bill_days, bill_amounts, bill_expiry_dates):
            if current_date.day == int(day):
                if expiry == "null" or current_date.strftime("%Y-%m-%d") <= expiry:
                    balance -= amount

        # Apply ad hoc cashflows
        for date, amount in zip(cashflow_dates, cashflow_amounts):
            if current_date.strftime("%Y-%m-%d") == date:
                balance += amount

        # Increment the date by one day and add balance (rounded to 2) and date to the list
        balances.append(round(balance, 2))
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)

    # Find local maxima and minima in all balances
    local_maxima = [
        (dates[i], balances[i])
        for i in range(3, len(balances) - 3)
        if balances[i] == max(balances[i - 3 : i + 4])
        and balances[i] != balances[i - 1]
    ]
    local_minima = [
        (dates[i], balances[i])
        for i in range(3, len(balances) - 3)
        if balances[i] == min(balances[i - 3 : i + 4])
        and balances[i] != balances[i - 1]
    ]

    return dedent(
        f"""
    Local Minima: {local_minima},
    Local Maxima: {local_maxima}
    """
    )


@https_fn.on_request(memory=options.MemoryOption.GB_1)
def https_function(req: https_fn.Request):
    """Main entry point for the application."""
    with app.request_context(req.environ):
        return app.full_dispatch_request()
