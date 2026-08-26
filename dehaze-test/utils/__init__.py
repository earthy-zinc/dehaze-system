"""Dehaze 联调测试工具库。

模块：
- config: 三端后端、Redis、MySQL 连接配置
- redis:  Redis 直连单例（不依赖 docker）
- mysql:  pymysql 直连 + 业务便捷查询
- auth:   多后端、多用户登录 + session 缓存
- api:    httpx 客户端，自动注入 X-Session-Id
- sse:    SSE 流式请求客户端（自动 Idempotency-Key + 标准 SSE 分行解析）
- cleanup: 限流 / 验证码 / session 清理
"""
