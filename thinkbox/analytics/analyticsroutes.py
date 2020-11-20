from . import ana
from flask_jwt_extended import jwt_required, get_jwt_identity
from flask import request, jsonify, Response, session
from thinkbox.models.uploadmodels import Uploads, UploadsSchema
from thinkbox.utils.analytics import *
from thinkbox.utils.dataupdate import datatype_update, preprocess_data
from thinkbox.utils.regression import *
import pandas as pd


# To load the particular data for preprocessing and analytics.
@ana.route("/load", methods=["GET"])
@jwt_required
def load_data():
    file_id = request.form['fileid']
    current_user = get_jwt_identity()
    file = Uploads.query.filter_by(id=file_id).first()
    if not file or not file.user_id == current_user['id']:
        return Response(response="error", status=204, content_type="application/json")
    session['data'] = pd.read_csv(file.filepath)
    session['loaded'] = True
    session['preprocessed'] = False
    return jsonify(UploadsSchema().dump(file)), 200


# View the data along with the types to check if the data is inferred properly
@ana.route("/view", methods=["GET"])
@jwt_required
def view():
    if session['loaded']:
        viewhead = request.form['head']
        if not bool(viewhead):
            resp = Response(response=session['data'].to_json(orient='records'), status=200,
                            content_type="application/json")
            return resp
        resp = Response(response=session['data'].head(int(viewhead)).to_json(orient='records'),
                        status=200,
                        content_type="application/json")
        return resp
    else:
        return Response(response={"message": "data not loaded"}, status=204, content_type='application/json')


# Updating the schema of the data if there were any changes in the data.
@ana.route("/update", methods=["POST"])
@jwt_required
def update_data():
    if session['loaded']:
        df = session['data']  # loading the data into the dataframe
        updates = request.get_json()
        df = datatype_update(df, updates)
        session['data'] = df
        return Response(response=session['data'].to_json(orient='records'), status=200, content_type="application/json")
    else:
        return Response(response={"message": "data not loaded"}, status=204, content_type='application/json')


# Preprocessing the data
@ana.route("/preprocess", methods=["GET"])
@jwt_required
def preprocess():
    target = request.get_json()['target']
    style = request.get_json()['style']
    bins = request.get_json()['bins']
    labels = request.get_json()['labels']
    droppable = request.get_json()['drop']
    df = session['data']  # Loading the data into dataframe
    if not style:
        df = preprocess_data(df, target, droppable)
    else:
        df = preprocess_data(df, target, droppable, style, bins, labels)
    session['data'] = df
    session['target'] = target
    session['preprocessed'] = True
    return Response(response=session['data'].to_json(orient='records'), status=200, content_type='application/json')


# Correlation Matrix
@ana.route("/correlation")
@jwt_required
def correlation():
    if session['preprocessed']:
        df = session['data']
        method = request.get_json()['method']
        session['corr_matrix'] = corr(df, method)
        resp = Response(response=session['corr_matrix'].to_json(orient='records'), status=200,
                        content_type='application/json')
        return resp
    else:
        resp = Response(response={'please preprocess the data'}, status=200, content_type="application/json")
        return resp


# Prediction Power
@ana.route("/prediction-power")
@jwt_required
def pp_score():
    if session['preprocessed']:
        df = session['data']
        target = session['target']
        session['pred_power'] = pred_power(df, target)
        resp = Response(response=session['pred_power'].to_json(orient='records'), status=200,
                        content_type='application/json')
        return resp
    else:
        resp = Response(response={'message': 'please preprocess the data'}, status=200, content_type="application/json")
        return resp


# Conduct Statistical Test
@ana.route("/test")
@jwt_required
def test():
    if session['preprocessed']:
        target = session['target']  # Getting our target feature from the session storage
        df = session['data']  # Loading the preprocessed dataframe
        test_results = conduct_test(df, target)
        tr_data = test_results.T
        significant_cols = tr_data[tr_data['test decision'] == 'significant'].index.to_list()
        session['significant_cols'] = significant_cols
        session['num_cols'] = df[significant_cols].select_dtypes(exclude='category').columns.to_list()
        session['cat_cols'] = df[significant_cols].select_dtypes('category').columns.to_list()
        return Response(response=test_results.to_json(), status=200,
                        content_type="application/json")
    else:
        return Response(response={"message": "data not loaded"}, status=204, content_type='application/json')


@ana.route("model-analytics", methods=['GET'])
@jwt_required
def model_analytics():
    # Current Version of API's Regression models
    regression_models = {
        'Linear Regression': linear_regression,
        'Decision Tree': decision_tree,
        'Random Forest': random_forest,
        'KNN Regression': knn_regression,
        'Ada Boost': ada_boost,
        'Gradient Boost': gradient_boost,
        'Polynomial Regression': polynomial_regression,
        'Elastic Net': elastic_net,
        'Ridge Regression': ridge_regression,
        'Lasso Regression': lasso_regression,
        'Light GBM': lightgbm,
        'XGB Regression': xgb_regression,
    }
    models = request.get_json()['models']
    models = models if bool(models) else regression_models.keys()
    test_results = {}
    df = session['data']
    significant_cols = session['significant_cols']
    target = session['target']
    cat_cols = session['cat_cols']
    num_cols = session['num_cols']
    for model in models:
        test_results[model] = {}
        test_results[model]['r2 score'], test_results[model]['rmse'], test_results[model]['model_params'] = \
            regression_models[model](df, significant_cols,
                                     target, num_cols,
                                     cat_cols)
    model_info = pd.DataFrame(test_results)
    resp = Response(response=model_info.to_json(orient='records'), status=200, content_type="application/json")
    return resp
