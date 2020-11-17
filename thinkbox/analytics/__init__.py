from flask import Blueprint

ana = Blueprint("ana", __name__)

from . import analyticsroutes
