from datetime import time
import os
from flask import current_app

def generate_predict_img(id):
    name = "pred_" + id + "_" + str(int(round(time.time() * 1000))) + ".jpg"
    path = os.path.join(current_app.config['PREDICT_IMAGE_PATH'], name)
    url = current_app.config['BASE_URL'] + "/predict/" + name
    return {
        "name": name,
        "path": path,
        "url": url
    }
