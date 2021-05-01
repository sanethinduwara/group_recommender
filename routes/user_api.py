from flask import jsonify, Blueprint
from flask_cors import CORS

from service.user_service import UserService

user_api = Blueprint('user_api', __name__)
cors = CORS(user_api, headers="Content-Type")

user_service = UserService()


@user_api.route("/", methods=['GET'])
def create_group():
    return jsonify(user_service.get_users().to_dict('records')), 200
