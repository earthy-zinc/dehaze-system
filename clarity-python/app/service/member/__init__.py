"""会员域：会员档案/管理、成长值/签到、权益配置、配额、到期处理五个子域。

外部引用统一走模块路径：
`from app.service.member.<module> import <Service类>`。
子域间依赖单向：growth/benefit/quota/expiry → member_service（共享工具）。
"""
