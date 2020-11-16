from thinkbox import db


class RevokedToken(db.Model):
    __tablename__ = "revokedtokens"
    id = db.Column(db.Integer, nullable=False, primary_key=True)
    jti = db.Column(db.String(255), nullable=False, index=True, unique=True)
    revoked_date = db.Column(db.DateTime, nullable=False)
