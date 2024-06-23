from flasgger import swag_from

from base import base as app
from flask import request, send_from_directory

from global_variable import DATA_PATH


@app.route('/predict/<filepath>', methods=['GET'])
@swag_from({
    'tags': ['10.模型接口'],
    'summary': '获取预测图片',
    'parameters': [
        {
            'name': 'filepath',
            'in': 'path',
            'type': 'string',
            'required': True
        }
    ],
    'responses': {
        200: {
            'description': '成功获取到图片文件',
            'content': {
                'application/octet-stream': {
                    'schema': {
                        'type': 'string',
                        'format': 'binary'
                    }
                }
            },
            'headers': {
                'Content-Disposition': {
                    'schema': {
                        'type': 'string',
                        'example': 'attachment; filename="example.jpg"'
                    }
                }
            }
        }
    }
})
def get_pred_image(filepath):
    return send_from_directory(DATA_PATH, filepath)
