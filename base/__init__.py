from flask import Blueprint
from flask_sqlalchemy import SQLAlchemy

base = Blueprint('base', __name__)
db = SQLAlchemy()

from route import *


