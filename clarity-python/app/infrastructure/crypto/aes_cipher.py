"""AES-256-CBC 加密/解密工具，用于 API Key 密文存储"""

import base64
import hashlib
import os

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from app.config import settings


def _derive_key() -> bytes:
    """由配置密钥派生 32 字节 AES 密钥"""
    return hashlib.sha256(settings.AI_PROVIDER_KEY_ENCRYPTION_KEY.encode()).digest()


def encrypt(plaintext: str) -> str:
    """AES-256-CBC 加密，返回 base64 编码的密文（iv + ct）"""
    key = _derive_key()
    iv = os.urandom(16)
    padder = padding.PKCS7(128).padder()
    padded = padder.update(plaintext.encode()) + padder.finalize()
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    encryptor = cipher.encryptor()
    ct = encryptor.update(padded) + encryptor.finalize()
    return base64.b64encode(iv + ct).decode()


def decrypt(ciphertext: str) -> str:
    """AES-256-CBC 解密"""
    key = _derive_key()
    raw = base64.b64decode(ciphertext)
    iv, ct = raw[:16], raw[16:]
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    padded = decryptor.update(ct) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    return (unpadder.update(padded) + unpadder.finalize()).decode()


def hash_key(plaintext: str) -> str:
    """SHA256 哈希（用于查重）"""
    return hashlib.sha256(plaintext.encode()).hexdigest()


def mask_key(plaintext: str) -> str:
    """截取前 8 位 + ...（展示用）"""
    return plaintext[:8] + "..." if len(plaintext) > 8 else plaintext + "..."
