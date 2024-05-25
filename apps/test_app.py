from flask import Flask

from apps import app


@app.route('/')
def hello_world():
    return 'Hello World!'
