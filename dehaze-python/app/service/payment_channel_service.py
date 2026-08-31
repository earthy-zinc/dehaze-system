"""支付渠道对接服务

通过配置控制启用：PAYMENT_WECHAT_ENABLED / PAYMENT_ALIPAY_ENABLED
- 未启用时降级为 mock（保持当前行为），启用时调用真实渠道 REST API
- 使用 httpx 直接调用渠道 REST API，不引入额外 SDK
"""

import base64
import json
import logging
import secrets
import time

import httpx

from app.config import settings
from app.core.code import ResultCode
from app.core.exceptions import BusinessException

logger = logging.getLogger(__name__)


class PayResult:
    def __init__(
        self, pay_url: str, channel_order_no: str | None = None, qr_code: str | None = None
    ):
        self.pay_url = pay_url
        self.channel_order_no = channel_order_no
        self.qr_code = qr_code or pay_url

    def to_dict(self) -> dict:
        return {
            "payUrl": self.pay_url,
            "qrCode": self.qr_code,
            "channelOrderNo": self.channel_order_no,
        }


class CallbackResult:
    def __init__(
        self, order_no: str, channel_payment_no: str, amount: int, success: bool, raw: dict
    ):
        self.order_no = order_no
        self.channel_payment_no = channel_payment_no
        self.amount = amount
        self.success = success
        self.raw = raw


class RefundResult:
    def __init__(
        self, channel_refund_no: str, success: bool, raw: dict, error_message: str | None = None
    ):
        self.channel_refund_no = channel_refund_no
        self.success = success
        self.raw = raw
        self.error_message = error_message


class BasePaymentChannel:
    channel: str = ""

    async def unified_order(self, order_no: str, amount: int, description: str) -> PayResult:
        raise NotImplementedError

    async def verify_callback(self, headers: dict, body: bytes) -> CallbackResult:
        raise NotImplementedError

    async def refund(
        self, order_no: str, channel_payment_no: str, refund_amount: int, total_amount: int
    ) -> RefundResult:
        raise NotImplementedError

    async def close_order(self, order_no: str) -> None:
        raise NotImplementedError

    async def download_bill(self, bill_date) -> list[dict] | None:
        """下载渠道对账账单，返回 [{orderNo, paymentNo, amount}]；未接入返回 None。"""
        return None


class MockPaymentChannel(BasePaymentChannel):
    channel = "mock"

    async def unified_order(self, order_no: str, amount: int, description: str) -> PayResult:
        pay_url = f"https://mock-pay.example.com/{order_no}"
        return PayResult(pay_url=pay_url, qr_code=pay_url, channel_order_no=f"MOCK{order_no}")

    async def verify_callback(self, headers: dict, body: bytes) -> CallbackResult:
        try:
            data = json.loads(body)
        except Exception:
            data = {}
        order_no = data.get("out_trade_no", "")
        channel_payment_no = data.get("transaction_id", f"MOCKPAY{order_no}")
        amount = int(data.get("amount", 0))
        return CallbackResult(
            order_no=order_no,
            channel_payment_no=channel_payment_no,
            amount=amount,
            success=data.get("trade_state", "SUCCESS") == "SUCCESS",
            raw=data,
        )

    async def refund(
        self, order_no: str, channel_payment_no: str, refund_amount: int, total_amount: int
    ) -> RefundResult:
        refund_no = f"MOCKRF{order_no}{int(time.time())}"
        return RefundResult(channel_refund_no=refund_no, success=True, raw={"refund_no": refund_no})

    async def close_order(self, order_no: str) -> None:
        return None


class WechatPayService(BasePaymentChannel):
    channel = "wechat"

    def __init__(self):
        self.app_id = settings.PAYMENT_WECHAT_APP_ID
        self.mch_id = settings.PAYMENT_WECHAT_MCH_ID
        self.api_v3_key = settings.PAYMENT_WECHAT_API_V3_KEY
        self.cert_serial_no = settings.PAYMENT_WECHAT_CERT_SERIAL_NO
        self.private_key_path = settings.PAYMENT_WECHAT_PRIVATE_KEY_PATH
        self.notify_url = settings.PAYMENT_WECHAT_NOTIFY_URL
        self.base_url = settings.PAYMENT_WECHAT_BASE_URL
        self._private_key = None

    def _get_private_key(self):
        if self._private_key is None:
            import os

            if not self.private_key_path or not os.path.exists(self.private_key_path):
                raise BusinessException(
                    ResultCode.SYSTEM_EXECUTION_ERROR, "微信支付私钥文件未配置或不存在"
                )
            from cryptography.hazmat.primitives import serialization

            with open(self.private_key_path, "rb") as f:
                self._private_key = serialization.load_pem_private_key(f.read(), password=None)
        return self._private_key

    def _build_authorization(self, method: str, url: str, body: str) -> str:
        timestamp = str(int(time.time()))
        nonce = secrets.token_hex(16)
        message = f"{self.mch_id}\n{nonce}\n{timestamp}\n{method}\n{url}\n{body}\n"
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        signature = self._get_private_key().sign(
            message.encode("utf-8"),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        sign_b64 = base64.b64encode(signature).decode("utf-8")
        return (
            f'WECHATPAY2-SHA256-RSA2048 mchid="{self.mch_id}",'
            f'nonce_str="{nonce}",timestamp="{timestamp}",'
            f'serial_no="{self.cert_serial_no}",signature="{sign_b64}"'
        )

    async def unified_order(self, order_no: str, amount: int, description: str) -> PayResult:
        path = "/v3/pay/transactions/native"
        payload = {
            "appid": self.app_id,
            "mchid": self.mch_id,
            "description": description,
            "out_trade_no": order_no,
            "notify_url": self.notify_url,
            "amount": {"total": amount, "currency": "CNY"},
        }
        body_str = json.dumps(payload, separators=(",", ":"))
        auth = self._build_authorization("POST", path, body_str)
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self.base_url}{path}",
                content=body_str,
                headers={
                    "Authorization": auth,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            if resp.status_code != 200:
                logger.error(
                    "微信下单失败 orderNo=%s status=%s body=%s",
                    order_no,
                    resp.status_code,
                    resp.text,
                )
                raise BusinessException(
                    ResultCode.CALL_THIRD_PARTY_SERVICE_ERROR, f"微信下单失败: {resp.text}"
                )
            data = resp.json()
            code_url = data.get("code_url", "")
            return PayResult(pay_url=code_url, qr_code=code_url, channel_order_no=None)

    async def verify_callback(self, headers: dict, body: bytes) -> CallbackResult:
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            raise BusinessException(
                ResultCode.SYSTEM_EXECUTION_ERROR, "缺少 cryptography 依赖"
            ) from None

        timestamp = headers.get("wechatpay-timestamp", "")
        nonce = headers.get("wechatpay-nonce", "")
        signature_b64 = headers.get("wechatpay-signature", "")
        serial_no = headers.get("wechatpay-serial", "")

        body_str = body.decode("utf-8") if isinstance(body, bytes) else body
        auth_message = f"{timestamp}\n{nonce}\n{body_str}\n"

        if not self._verify_wechat_signature(auth_message, signature_b64, serial_no):
            raise BusinessException(ResultCode.PARAM_ERROR, "微信回调验签失败")

        payload = json.loads(body_str)
        resource = payload.get("resource", {})
        ciphertext = resource.get("ciphertext", "")
        nonce_str = resource.get("nonce", "")
        associated_data = resource.get("associated_data", "")

        key = self.api_v3_key.encode("utf-8")
        aesgcm = AESGCM(key)
        ciphertext_bytes = base64.b64decode(ciphertext)
        decrypted = aesgcm.decrypt(
            ciphertext_bytes, nonce_str.encode("utf-8"), associated_data.encode("utf-8")
        )
        data = json.loads(decrypted.decode("utf-8"))

        return CallbackResult(
            order_no=data.get("out_trade_no", ""),
            channel_payment_no=data.get("transaction_id", ""),
            amount=int(data.get("amount", {}).get("total", 0)),
            success=data.get("trade_state") == "SUCCESS",
            raw=data,
        )

    def _verify_wechat_signature(self, message: str, signature_b64: str, serial_no: str) -> bool:
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding
            from cryptography.x509 import load_pem_x509_certificate
        except ImportError:
            raise BusinessException(
                ResultCode.SYSTEM_EXECUTION_ERROR, "缺少 cryptography 依赖"
            ) from None

        cert_text = self._get_platform_cert(serial_no)
        cert = load_pem_x509_certificate(cert_text.encode("utf-8"))
        pub_key = cert.public_key()
        signature = base64.b64decode(signature_b64)
        try:
            pub_key.verify(
                signature,
                message.encode("utf-8"),
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
            return True
        except Exception:
            return False

    def _get_platform_cert(self, serial_no: str) -> str:
        import os

        cert_cache_dir = os.path.join(settings.TEMP_DIR_RESOLVED, "wechat_certs")
        os.makedirs(cert_cache_dir, exist_ok=True)
        cert_path = os.path.join(cert_cache_dir, f"{serial_no}.pem")
        if os.path.exists(cert_path):
            with open(cert_path, encoding="utf-8") as f:
                return f.read()
        asyncio = __import__("asyncio")
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            raise BusinessException(
                ResultCode.SYSTEM_EXECUTION_ERROR, f"微信平台证书未缓存 serial={serial_no}"
            )
        import httpx as _httpx

        resp = _httpx.get(
            f"{self.base_url}/v3/certificates", headers={"Accept": "application/json"}
        )
        if resp.status_code != 200:
            raise BusinessException(
                ResultCode.CALL_THIRD_PARTY_SERVICE_ERROR, "获取微信平台证书失败"
            )
        certs = resp.json().get("data", [])
        for c in certs:
            if c.get("serial_no") == serial_no:
                cert_b64 = c.get("encrypt_certificate", {}).get("ciphertext", "")
                cert_text = base64.b64decode(cert_b64).decode("utf-8")
                with open(cert_path, "w", encoding="utf-8") as f:
                    f.write(cert_text)
                return cert_text
        raise BusinessException(
            ResultCode.SYSTEM_EXECUTION_ERROR, f"未找到微信平台证书 serial={serial_no}"
        )

    async def refund(
        self, order_no: str, channel_payment_no: str, refund_amount: int, total_amount: int
    ) -> RefundResult:
        path = "/v3/refund/domestic/refunds"
        refund_no = f"RF{order_no}{int(time.time())}"
        payload = {
            "out_trade_no": order_no,
            "out_refund_no": refund_no,
            "reason": "用户申请退款",
            "amount": {
                "refund": refund_amount,
                "total": total_amount,
                "currency": "CNY",
            },
            "notify_url": settings.PAYMENT_WECHAT_REFUND_NOTIFY_URL,
        }
        body_str = json.dumps(payload, separators=(",", ":"))
        auth = self._build_authorization("POST", path, body_str)
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self.base_url}{path}",
                content=body_str,
                headers={
                    "Authorization": auth,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            if resp.status_code != 200:
                logger.error(
                    "微信退款失败 orderNo=%s status=%s body=%s",
                    order_no,
                    resp.status_code,
                    resp.text,
                )
                return RefundResult(
                    channel_refund_no=refund_no,
                    success=False,
                    raw={},
                    error_message=resp.text,
                )
            data = resp.json()
            success = data.get("status") in ("SUCCESS", "PROCESSING")
            return RefundResult(
                channel_refund_no=data.get("refund_id", refund_no),
                success=success,
                raw=data,
                error_message=None if success else data.get("status"),
            )

    async def close_order(self, order_no: str) -> None:
        path = f"/v3/pay/transactions/out-trade-no/{order_no}/close"
        payload = {"mchid": self.mch_id}
        body_str = json.dumps(payload, separators=(",", ":"))
        auth = self._build_authorization("POST", path, body_str)
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{self.base_url}{path}",
                content=body_str,
                headers={
                    "Authorization": auth,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            if resp.status_code not in (200, 204):
                logger.warning("微信关单失败 orderNo=%s status=%s", order_no, resp.status_code)


class AlipayService(BasePaymentChannel):
    channel = "alipay"

    def __init__(self):
        self.app_id = settings.PAYMENT_ALIPAY_APP_ID
        self.private_key = settings.PAYMENT_ALIPAY_PRIVATE_KEY
        self.public_key = settings.PAYMENT_ALIPAY_PUBLIC_KEY
        self.notify_url = settings.PAYMENT_ALIPAY_NOTIFY_URL
        self.base_url = settings.PAYMENT_ALIPAY_BASE_URL

    def _sign(self, params: dict) -> str:
        sorted_items = sorted([(k, v) for k, v in params.items() if v is not None and v != ""])
        sign_str = "&".join(f"{k}={v}" for k, v in sorted_items)
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding

        private_key_obj = serialization.load_pem_private_key(
            self.private_key.encode("utf-8"), password=None
        )
        signature = private_key_obj.sign(
            sign_str.encode("utf-8"),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _verify(self, params: dict, sign: str) -> bool:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding

        sorted_items = sorted(
            [
                (k, v)
                for k, v in params.items()
                if k != "sign" and k != "sign_type" and v is not None and v != ""
            ]
        )
        sign_str = "&".join(f"{k}={v}" for k, v in sorted_items)
        public_key_obj = serialization.load_pem_public_key(self.public_key.encode("utf-8"))
        try:
            public_key_obj.verify(
                base64.b64decode(sign),
                sign_str.encode("utf-8"),
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
            return True
        except Exception:
            return False

    def _build_biz_content(self, biz: dict) -> str:
        return json.dumps(biz, separators=(",", ":"), ensure_ascii=False)

    async def unified_order(self, order_no: str, amount: int, description: str) -> PayResult:
        biz_content = self._build_biz_content(
            {
                "out_trade_no": order_no,
                "total_amount": f"{amount / 100:.2f}",
                "subject": description,
                "product_code": "FACE_TO_FACE_PAYMENT",
            }
        )
        params = {
            "app_id": self.app_id,
            "method": "alipay.trade.precreate",
            "charset": "utf-8",
            "sign_type": "RSA2",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "1.0",
            "biz_content": biz_content,
            "notify_url": self.notify_url,
        }
        params["sign"] = self._sign(params)
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(self.base_url, data=params)
            if resp.status_code != 200:
                logger.error("支付宝下单失败 orderNo=%s status=%s", order_no, resp.status_code)
                raise BusinessException(ResultCode.CALL_THIRD_PARTY_SERVICE_ERROR, "支付宝下单失败")
            data = resp.json()
            resp_data = data.get("alipay_trade_precreate_response", {})
            if resp_data.get("code") != "10000":
                logger.error("支付宝下单失败 orderNo=%s body=%s", order_no, data)
                raise BusinessException(
                    ResultCode.CALL_THIRD_PARTY_SERVICE_ERROR,
                    f"支付宝下单失败: {resp_data.get('sub_msg')}",
                )
            qr_code = resp_data.get("qr_code", "")
            return PayResult(pay_url=qr_code, qr_code=qr_code, channel_order_no=None)

    async def verify_callback(self, headers: dict, body: bytes) -> CallbackResult:
        from urllib.parse import parse_qs

        body_str = body.decode("utf-8") if isinstance(body, bytes) else body
        params = dict(parse_qs(body_str, keep_blank_values=True))
        params = {k: v[0] if isinstance(v, list) and v else v for k, v in params.items()}
        sign = params.pop("sign", "")
        params.pop("sign_type", None)
        if not self._verify(params, sign):
            raise BusinessException(ResultCode.PARAM_ERROR, "支付宝回调验签失败")
        trade_status = params.get("trade_status", "")
        amount_str = params.get("total_amount", "0")
        amount = int(float(amount_str) * 100)
        return CallbackResult(
            order_no=params.get("out_trade_no", ""),
            channel_payment_no=params.get("trade_no", ""),
            amount=amount,
            success=trade_status in ("TRADE_SUCCESS", "TRADE_FINISHED"),
            raw=params,
        )

    async def refund(
        self, order_no: str, channel_payment_no: str, refund_amount: int, total_amount: int
    ) -> RefundResult:
        refund_no = f"RF{order_no}{int(time.time())}"
        biz_content = self._build_biz_content(
            {
                "out_trade_no": order_no,
                "trade_no": channel_payment_no,
                "refund_amount": f"{refund_amount / 100:.2f}",
                "out_request_no": refund_no,
            }
        )
        params = {
            "app_id": self.app_id,
            "method": "alipay.trade.refund",
            "charset": "utf-8",
            "sign_type": "RSA2",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "1.0",
            "biz_content": biz_content,
        }
        params["sign"] = self._sign(params)
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(self.base_url, data=params)
            if resp.status_code != 200:
                return RefundResult(
                    channel_refund_no=refund_no,
                    success=False,
                    raw={},
                    error_message=f"HTTP {resp.status_code}",
                )
            data = resp.json()
            resp_data = data.get("alipay_trade_refund_response", {})
            if resp_data.get("code") != "10000":
                return RefundResult(
                    channel_refund_no=refund_no,
                    success=False,
                    raw=resp_data,
                    error_message=resp_data.get("sub_msg"),
                )
            return RefundResult(
                channel_refund_no=resp_data.get("trade_no", refund_no),
                success=True,
                raw=resp_data,
            )

    async def close_order(self, order_no: str) -> None:
        biz_content = self._build_biz_content({"out_trade_no": order_no})
        params = {
            "app_id": self.app_id,
            "method": "alipay.trade.close",
            "charset": "utf-8",
            "sign_type": "RSA2",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "1.0",
            "biz_content": biz_content,
        }
        params["sign"] = self._sign(params)
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(self.base_url, data=params)
            if resp.status_code != 200:
                logger.warning("支付宝关单失败 orderNo=%s status=%s", order_no, resp.status_code)


class PaymentChannelService:
    """支付渠道统一入口，按 channel 路由到具体渠道服务"""

    def __init__(self):
        self._channels: dict[str, BasePaymentChannel] = {}
        self._init_channels()

    def _init_channels(self):
        if settings.PAYMENT_WECHAT_ENABLED:
            self._channels["wechat"] = WechatPayService()
        if settings.PAYMENT_ALIPAY_ENABLED:
            self._channels["alipay"] = AlipayService()
        self._mock = MockPaymentChannel()

    def get_channel(self, channel: str) -> BasePaymentChannel:
        if channel not in ("wechat", "alipay"):
            raise BusinessException(ResultCode.PARAM_ERROR, f"不支持的支付渠道: {channel}")
        return self._channels.get(channel, self._mock)

    async def unified_order(
        self, channel: str, order_no: str, amount: int, description: str
    ) -> PayResult:
        return await self.get_channel(channel).unified_order(order_no, amount, description)

    async def verify_callback(self, channel: str, headers: dict, body: bytes) -> CallbackResult:
        return await self.get_channel(channel).verify_callback(headers, body)

    async def refund(
        self,
        channel: str,
        order_no: str,
        channel_payment_no: str,
        refund_amount: int,
        total_amount: int,
    ) -> RefundResult:
        return await self.get_channel(channel).refund(
            order_no, channel_payment_no, refund_amount, total_amount
        )

    async def close_order(self, channel: str, order_no: str) -> None:
        return await self.get_channel(channel).close_order(order_no)

    async def download_bill(self, channel: str, bill_date) -> list[dict] | None:
        """渠道账单下载：渠道未启用（走 mock）或未实现账单能力时返回 None，跳过对账。"""
        impl = self._channels.get(channel)
        if impl is None:
            return None
        return await impl.download_bill(bill_date)


payment_channel_service = PaymentChannelService()
