from common import success
from base import base as app

@app.route('/', methods=['GET'])
def hello_world():
    return success('Hello World!')
