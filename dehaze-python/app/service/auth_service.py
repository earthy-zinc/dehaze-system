import uuid
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from app.utils.jwt_util import jwt_required, get_current_user_id
import random
import string
import jwt
from datetime import datetime, timedelta
from flask import current_app, request
from app.models import SysUser
from app.extensions import mysql


class AuthService:
    """认证服务类"""

    @staticmethod
    def login(username: str, password: str) -> dict:
        """
        用户登录
        
        Args:
            username (str): 用户名
            password (str): 密码
            
        Returns:
            dict: 登录结果
        """
        # 验证用户凭据
        user = SysUser.query.filter_by(username=username, deleted=0).first()
        
        if not user or not user.check_password(password):
            raise Exception("用户名或密码错误")
            
        if user.status != 1:
            raise Exception("用户已被禁用")
        
        # 生成访问令牌
        payload = {
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(hours=24),
            'iat': datetime.utcnow()
        }
        access_token = jwt.encode(
            payload, 
            current_app.config.get('SECRET_KEY', 'default_secret_key'), 
            algorithm='HS256'
        )
        
        return {
            'tokenType': 'Bearer',
            'accessToken': access_token
        }

    @staticmethod
    def logout():
        """
        用户注销
        """
        # 获取当前用户的ID
        user_id = get_current_user_id()
        if not user_id:
            raise Exception("未找到有效的用户会话")
        
        # 在实际应用中，可以将令牌加入黑名单
        # 这里简化处理，仅清除会话相关信息
        pass

    @staticmethod
    def get_captcha() -> dict:
        """
        获取验证码
        
        Returns:
            dict: 验证码信息
        """
        # 生成验证码文本
        captcha_text = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
        
        # 生成验证码图片
        image = Image.new('RGB', (120, 40), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # 使用默认字体绘制文本
        try:
            font = ImageFont.truetype('arial.ttf', 24)
        except:
            font = ImageFont.load_default()
            
        draw.text((20, 10), captcha_text, fill=(0, 0, 0), font=font)
        
        # 添加一些干扰线
        for _ in range(5):
            x1 = random.randint(0, 120)
            y1 = random.randint(0, 40)
            x2 = random.randint(0, 120)
            y2 = random.randint(0, 40)
            draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 0), width=1)
        
        # 将图片转换为base64编码
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # 生成验证码key
        captcha_key = str(uuid.uuid4())
        
        # 将验证码文本存储到Redis中，设置过期时间（5分钟）
        redis_client = current_app.extensions.get('redis_client')
        if redis_client:
            redis_client.setex(f"captcha:{captcha_key}", 300, captcha_text)
        
        return {
            'captchaKey': captcha_key,
            'captchaBase64': f"data:image/jpeg;base64,{img_str}"
        }

    @staticmethod
    def verify_captcha(captcha_key: str, captcha_code: str) -> bool:
        """
        验证验证码
        
        Args:
            captcha_key (str): 验证码key
            captcha_code (str): 用户输入的验证码
            
        Returns:
            bool: 验证结果
        """
        # 获取Redis客户端
        redis_client = current_app.extensions.get('redis_client')
        
        # 检查Redis客户端是否存在
        if not redis_client:
            return False
            
        # 从Redis中获取验证码文本
        stored_captcha = redis_client.get(f"captcha:{captcha_key}")
        
        if not stored_captcha:
            return False
            
        # 比较验证码（不区分大小写）
        result = stored_captcha.decode().lower() == captcha_code.lower()
        
        # 验证后删除验证码
        if result:
            redis_client.delete(f"captcha:{captcha_key}")
            
        return result