from flask_jwt_extended import jwt_required, get_jwt_identity
from flask import request, jsonify
from werkzeug.utils import secure_filename
from . import dash
import os
from config import UPLOADS_PATH
from thinkbox.models.uploadmodels import Uploads, UploadsSchema
from thinkbox import db
from datetime import datetime

ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@dash.route("/upload", methods=["POST"])
@jwt_required
def upload():
    if 'file' not in request.files:
        return jsonify(message="No file Detected")
    file = request.files['file']
    if file.filename and allowed_file(file.filename):
        if not os.path.isdir(os.path.abspath(
                UPLOADS_PATH + str(get_jwt_identity()['id']))):  # Check if the directory is present or not
            os.makedirs(os.path.abspath(UPLOADS_PATH + str(get_jwt_identity()['id'])))
        path = UPLOADS_PATH + str(get_jwt_identity()['id'])
        filename = secure_filename(file.filename)
        file.save(os.path.join(path, filename))
        uploaded_file = Uploads(filepath=os.path.join(path, filename), filename=filename,
                                uploaded_date=datetime.now(), user_id=get_jwt_identity()['id'])
        db.session.add(uploaded_file)
        db.session.commit()
        return jsonify({
            "message": "success",
            "path": os.path.join(path, filename)
        })


@dash.route("/files", methods=["GET"])
@jwt_required
def allfiles():
    # Show files from specific user.
    user_details = get_jwt_identity()
    file_details = Uploads.query.filter_by(user_id=user_details['id']).all()
    files = UploadsSchema().dump(file_details, many=True)
    return jsonify(files)


@dash.route("/", methods=["GET"])
@jwt_required
def index():
    # Just to check random stuff xD!!
    return "Index"
