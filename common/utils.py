import hashlib
from io import BytesIO

def calculate_file_md5(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def calculate_bytes_md5(bytes_io: BytesIO) -> str:
    hash_md5 = hashlib.md5()
    # 重置 BytesIO 对象的指针到开头
    bytes_io.seek(0)
    while True:
        chunk = bytes_io.read(4096)
        if not chunk:
            break
        hash_md5.update(chunk)
    return hash_md5.hexdigest()
