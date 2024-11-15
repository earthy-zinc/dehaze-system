from minio import Minio

def init_minio(app):
    minio_client = Minio(
        app.config.get("MINIO_ENDPOINT"),
        access_key=app.config.get("MINIO_ACCESS_KEY"),
        secret_key=app.config.get("MINIO_SECRET_KEY"),
        secure=app.config.get("MINIO_SECURE")
    )
    minio_bucket_name = app.config.get("MINIO_BUCKET_NAME")
    found = minio_client.bucket_exists(minio_bucket_name)
    if not found:
        minio_client.make_bucket(minio_bucket_name)
    return minio_client
