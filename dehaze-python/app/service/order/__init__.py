"""订单域：订单核心、支付、退款、自动续费四个子域。

外部引用统一走模块路径：
`from app.service.order.<module> import <service单例>`。
子域间依赖单向：payment/refund/auto_renew → order_service（共享工具），
auto_renew → payment_service（余额支付完成链路）。
"""
