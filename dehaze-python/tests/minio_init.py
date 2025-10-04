from minio import Minio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 创建Minio客户端
minio_client = Minio(
    "localhost:9000",
    access_key="your-access-key",
    secret_key="your-secret-key",
    secure=False
)

# 创建SQLAlchemy引擎
engine = create_engine("mysql+pymysql://username:password@localhost/database")
Session = sessionmaker(bind=engine)
session = Session()
