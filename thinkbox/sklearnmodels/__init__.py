from flask import Blueprint

skmodels = Blueprint("skmodels", __name__)

from . import sklearnmodelroutes
