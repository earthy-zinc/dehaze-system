"""测试桩包：按职责拆分为三个模块。

- fakes: 实现真实接口协议的仿真假件（DB/图/LLM/SSE/中断/仓储等）
- factories: 纯数据构造函数（redis/orm/conv/member/benefit/context 等）
- mocks: 对外围"外部世界"边界打桩的 monkeypatch 辅助
本包不做任何 re-export，使用方必须从具体子模块 import。
"""
