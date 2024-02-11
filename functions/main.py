"""This module is used as an entry point for the application.

This will house various cloud functions. 
"""

from firebase_admin import initialize_app
from firebase_functions import https_fn
import flask

initialize_app()
app = flask.Flask(__name__)


@app.get("/health")
def ping():
    """Health check endpoint."""
    return {"status": "ok"}


@https_fn.on_request()
def https_function(req: https_fn.Request):
    """Main entry point for the application."""
    with app.request_context(req.environ):
        return app.full_dispatch_request()
