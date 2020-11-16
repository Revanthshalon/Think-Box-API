from thinkbox import db, ma


class Uploads(db.Model):
    __tablename__ = "uploads"
    id = db.Column(db.Integer, primary_key=True)
    filepath = db.Column(db.String(255), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    uploaded_date = db.Column(db.DateTime, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)


class UploadsSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Uploads
        include_fk = True

        id = ma.auto_field()
        filepath = ma.auto_field()
        filename = ma.auto_field()
        uploaded_date = ma.auto_field()
