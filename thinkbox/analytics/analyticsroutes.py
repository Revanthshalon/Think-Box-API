from . import ana
from flask_jwt_extended import jwt_required, get_jwt_identity
from flask import request, jsonify, Response, session
from thinkbox.models.uploadmodels import Uploads, UploadsSchema
from thinkbox.utils.analytics import conduct_test
import pandas as pd


@ana.route("/load", methods=["GET"])
@jwt_required
def load_data():
    file_id = request.form['fileid']
    current_user = get_jwt_identity()
    file = Uploads.query.filter_by(id=file_id).first()
    if not file or not file.user_id == current_user['id']:
        return Response(response="error", status=204, content_type="application/json")
    session['data'] = pd.read_csv(file.filepath)
    return jsonify(UploadsSchema().dump(file)), 200


@ana.route("/view", methods=["GET"])
@jwt_required
def view():
    viewhead = request.form['head']
    if not bool(viewhead):
        resp = Response(response=session['data'].to_json(index=False, orient='table'), status=200,
                        content_type="application/json")
        return resp
    resp = Response(response=session['data'].head(int(viewhead)).to_json(index=False, orient='table'),
                    status=200,
                    content_type="application/json")
    return resp


@ana.route("/test")
@jwt_required
def test():
    droppable = request.form['drop']
    target = request.form['target']
    test_results = conduct_test(session['data'], target, droppable)
    return Response(response=test_results.to_json(), status=200,
                    content_type="application/json")
