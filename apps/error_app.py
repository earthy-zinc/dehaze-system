import json
from http.client import HTTPException

from apps import app


@app.errorhandler(HTTPException)
def handle_exception(e):
    response = e.get_response()
    response.data = json.dumps({
        'code': 0,
        'msg': e.description
    })
    response.content_type = 'application/json'
    return response
