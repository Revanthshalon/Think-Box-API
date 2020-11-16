from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager
from flask_marshmallow import Marshmallow
from flask_migrate import Migrate
import config as Config

db = SQLAlchemy()  # ORM Database
ma = Marshmallow()  # Marshmallow for serialization
jwt = JWTManager()  # Token Authentication


def create_app():
    # App Initialization
    app = Flask(__name__, instance_relative_config=True)

    # Configuration Settings
    Config.init_app(app)
    # Pluggable Flask Extension
    db.init_app(app)
    ma.init_app(app)
    Migrate(app, db)
    jwt.init_app(app)

    # Import Models
    from thinkbox.models import tokenmodels
    from thinkbox.models import uploadmodels
    from thinkbox.models import usermodels

    # Blueprints
    from thinkbox.auth import auth
    from thinkbox.dashboard import dash

    # Register Blueprints
    app.register_blueprint(auth, url_prefix="/auth")
    app.register_blueprint(dash, url_prefix="/dash")

    return app
