CREATE DATABASE IF NOT EXISTS ruoyi_vue_pro;
USE ruoyi_vue_pro;

CREATE STABLE IF NOT EXISTS iot_device_log (
    ts TIMESTAMP,
    id VARCHAR(36) PRIMARY KEY,
    request_id NCHAR(36),
    product_key NCHAR(64),
    device_name NCHAR(128),
    type NCHAR(32),
    identifier NCHAR(64),
    content NCHAR(1024),
    code INT
) TAGS (device_key NCHAR(128));
