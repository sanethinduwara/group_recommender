from flask import request, jsonify, Blueprint
from flask_cors import CORS, cross_origin

from service.group_service import GroupService

group_api = Blueprint('group_api', __name__)
cors = CORS(group_api, headers="Content-Type")

group_service = GroupService()


@group_api.route("/", methods=['POST'])
@cross_origin
def create_group():
    group_service.save_group(request.json)
    return jsonify(), 201
