from flask import request, jsonify
from flask_jwt_extended import create_access_token, create_refresh_token, jwt_refresh_token_required, get_jwt_identity, \
    jwt_required, get_raw_jwt
from sqlalchemy import exc

from thinkbox import db, jwt
from thinkbox.models.usermodels import User, UserSchema
from thinkbox.models.tokenmodels import RevokedToken
from . import auth
from datetime import datetime


@jwt.user_claims_loader
def add_claims_to_access_tokens(user):
    return {
        'user_role': user['role']
    }


@jwt.user_identity_loader
def add_identity_to_access_tokens(user):
    return user


@jwt.token_in_blacklist_loader
def check_if_token_in_blacklist(decrypted_token):
    jti = decrypted_token['jti']
    test = RevokedToken.query.filter_by(jti=jti).first()
    return bool(test)


@auth.route("/register", methods=["POST"])
def register():
    try:
        test = User.query.filter_by(email=request.form['email']).first()

        if test:
            return jsonify(message="User already Registered")

        user = User(
            first_name=request.form['firstname'],
            last_name=request.form['lastname'],
            middle_name=request.form['middlename'],
            email=request.form['email'],
            password=request.form['password'],
            role=1,
            created_date=datetime.now()
        )
        db.session.add(user)
        db.session.commit()

        return jsonify(message="successfully registered"), 200
    except exc:
        db.session.rollback()
        return jsonify(message="failed")


@auth.route("/login", methods=["POST"])
def login():
    user = User.query.filter_by(email=request.form['email']).first()
    if not user:
        return jsonify(message="user not found")
    elif user.verify_password(request.form['password']):
        user = UserSchema().dump(user)
        access_token = create_access_token(identity=user, fresh=True)
        refresh_token = create_refresh_token(identity=user)
        resp = {
            'access_token': access_token,
            'refresh_token': refresh_token
        }
        return jsonify(resp), 200
    else:
        return jsonify(message="Invalid Credentials"), 401


@auth.route("/refresh", methods=["POST"])
@jwt_refresh_token_required
def refresh():
    current_user = get_jwt_identity()
    access_token = create_access_token(current_user, fresh=False)
    resp = jsonify({
        "access_token": access_token
    })
    return resp, 200


@auth.route("/logout", methods=["DELETE"])
@jwt_required
def logout():
    jti = get_raw_jwt()['jti']
    rt = RevokedToken(jti=jti, revoked_date=datetime.now())
    db.session.add(rt)
    db.session.commit()
    return jsonify(message="Successfully Logged out"), 200


@auth.route("/logout2", methods=["DELETE"])
@jwt_refresh_token_required
def logout2():
    jti = get_raw_jwt()['jti']
    rt = RevokedToken(jti=jti, revoked_date=datetime.now())
    db.session.add(rt)
    db.session.commit()
    return jsonify(message="Successfully Logged out"), 200
