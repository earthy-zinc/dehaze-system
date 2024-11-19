import uuid
import os
import requests
from flask import request

from global_variable import DATA_PATH, CACHE_PATH


def generate_predict_img(id):
    name = "pred_" + id + "_" + str(uuid.uuid4()) + ".jpg"
    path = os.path.join(DATA_PATH, name)
    url = request.host_url + "/predict/" + name
    return {
        "name": name,
        "path": path,
        "url": url
    }


def generate_input_img(url, id):
    response = requests.get(url, stream=True)
    name = "input_" + id + "_" + str(uuid.uuid4()) + ".jpg"
    path = os.path.join(CACHE_PATH, name)
    if response.status_code == 200:
        with open(path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        return {
            "name": name,
            "path": path,
            "url": url
        }
    else:
        raise Exception("图片下载失败")


def get_image_bytes_from_url(url) -> BytesIO:
    try:
        urlparse(url)
        # 使用requests库获取图片
        response = requests.get(image_url, stream=True)
        response.raise_for_status()  # 确保请求成功

        # 将图片转换为BytesIO对象
        image_bytes = BytesIO(response.content)
        return image_bytes
    except ValueError:
    return none
