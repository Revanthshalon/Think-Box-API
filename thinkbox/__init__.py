from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager
from flask_marshmallow import Marshmallow
from flask_migrate import Migrate
from flask_session import Session
import config as Config

db = SQLAlchemy()  # ORM Database
ma = Marshmallow()  # Marshmallow for serialization
jwt = JWTManager()  # Token Authentication
sess = Session()  # Flask Session


def create_app():
    # App Initialization
    app = Flask(__name__, instance_relative_config=True)

    # Configuration Settings
    Config.init_app(app)
    # Pluggable Flask Extension
    db.init_app(app)
    ma.init_app(app)
    Migrate(app, db)
    sess.init_app(app)
    jwt.init_app(app)

    # Import Models
    from thinkbox.models import tokenmodels
    from thinkbox.models import uploadmodels
    from thinkbox.models import usermodels

    # Blueprints
    from thinkbox.auth import auth
    from thinkbox.dashboard import dash
    from thinkbox.analytics import ana
    from thinkbox.sklearnmodels import skmodels

    # Register Blueprints
    app.register_blueprint(auth, url_prefix="/auth")
    app.register_blueprint(dash, url_prefix="/dash")
    app.register_blueprint(ana, url_prefix="/dash/analytics")
    app.register_blueprint(skmodels, url_prefix="/dash/models")

    return app
