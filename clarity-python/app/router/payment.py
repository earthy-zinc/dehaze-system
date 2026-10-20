from fastapi import APIRouter, Depends, Request
from fastapi.responses import PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.service.order.payment_service import payment_service

router = APIRouter(
    prefix="/api/v1/orders/payment",
    tags=["支付回调"],
)


@router.post("/wechat/callback", summary="微信支付回调")
async def wechat_callback(request: Request, db: AsyncSession = Depends(get_db)):
    body = await request.body()
    headers = dict(request.headers)
    ok = await payment_service.handle_payment_callback(db, "wechat", headers, body)
    return {"code": "SUCCESS" if ok else "FAIL", "message": "成功" if ok else "失败"}


@router.post("/alipay/callback", summary="支付宝支付回调", response_class=PlainTextResponse)
async def alipay_callback(request: Request, db: AsyncSession = Depends(get_db)):
    body = await request.body()
    headers = dict(request.headers)
    ok = await payment_service.handle_payment_callback(db, "alipay", headers, body)
    return "success" if ok else "fail"
