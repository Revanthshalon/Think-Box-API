from thinkbox import db, ma
from werkzeug.security import generate_password_hash, check_password_hash


class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(100), nullable=False)
    middle_name = db.Column(db.String(100), nullable=True)
    last_name = db.Column(db.String(100), nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(100), unique=True, index=True, nullable=False)
    role = db.Column(db.SmallInteger, nullable=False)
    created_date = db.Column(db.DateTime, nullable=False)
    uploads = db.relationship('Uploads', backref='users', lazy=True)

    @property
    def password(self):
        raise AttributeError("Password is not a readable attribute")

    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)


class UserSchema(ma.SQLAlchemySchema):
    class Meta:  # Generating Schema for data serialization
        model = User

    id = ma.auto_field()
    email = ma.auto_field()
    role = ma.auto_field()
