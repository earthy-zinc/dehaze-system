from datetime import datetime, timedelta

from app.service.ai import memory_extraction as extraction
from app.service.ai import memory_injection as injection
from tests.stubs import make_orm_mem


def _mem(
    id_,
    type_,
    content,
    importance=50,
    last_accessed=None,
    create=None,
    metadata=None,
    relevance=1.0,
):
    return {
        "id": id_,
        "memory_type": type_,
        "content": content,
        "importance": importance,
        "last_accessed_at": last_accessed,
        "create_time": create or datetime.now(),
        "metadata": metadata,
        "relevance": relevance,
    }


def _install_injection_mocks(
    monkeypatch,
    *,
    prefs=None,
    skills=None,
    user_skills=None,
    es_retrievals=None,
    keyword_hits=None,
    touch_recorder=None,
):
    async def _list_preferences(db, user_id, limit=20):
        return prefs if prefs is not None else []

    async def _list_by_skill(db, user_id, skill, limit=10):
        return skills if skills is not None else []

    async def _list_skills(db, user_id):
        return user_skills if user_skills is not None else []

    async def _search_by_keyword(db, user_id, keyword, limit=5):
        return keyword_hits if keyword_hits is not None else []

    async def _search_memories(user_id, query, top_n=5):
        return es_retrievals if es_retrievals is not None else []

    async def _touch(db, mid):
        if touch_recorder is not None:
            touch_recorder.append(mid)

    monkeypatch.setattr(injection.ai_memory_repository, "list_preferences", _list_preferences)
    monkeypatch.setattr(injection.ai_memory_repository, "list_by_skill", _list_by_skill)
    monkeypatch.setattr(injection.ai_memory_repository, "list_skills", _list_skills)
    monkeypatch.setattr(injection.ai_memory_repository, "search_by_keyword", _search_by_keyword)
    monkeypatch.setattr(injection.ai_memory_repository, "touch", _touch)
    monkeypatch.setattr(injection, "search_memories", _search_memories)


class TestComputeImportance:

    def test_weighted_sum(self):
        factors = {
            "emotion": 100,
            "frequency": 100,
            "recency": 100,
            "novelty": 100,
            "explicit_mark": 0,
        }
        assert extraction._compute_importance(factors) == 90

    def test_explicit_mark_forces_100(self):
        factors = {
            "emotion": 10,
            "frequency": 10,
            "recency": 10,
            "novelty": 10,
            "explicit_mark": 100,
        }
        assert extraction._compute_importance(factors) == 100

    def test_all_zero(self):
        factors = {"emotion": 0, "frequency": 0, "recency": 0, "novelty": 0, "explicit_mark": 0}
        assert extraction._compute_importance(factors) == 0


class TestRecencyScore:

    def test_recent_is_near_1(self):
        score = injection._recency_score(datetime.now(), datetime.now())
        assert 0.99 < score <= 1.0

    def test_old_decays(self):
        old = datetime.now() - timedelta(days=30)
        score = injection._recency_score(old, old)
        assert 0.3 < score < 0.45

    def test_no_ts_uses_now(self):
        assert injection._recency_score(None, None) > 0.99


class TestSortRetrieved:

    def test_higher_relevance_ranks_first(self):
        now = datetime.now()
        mems = [
            _mem(1, "episodic", "A", importance=50, last_accessed=now, relevance=0.9),
            _mem(2, "episodic", "B", importance=50, last_accessed=now, relevance=0.2),
        ]
        result = injection._sort_retrieved(mems)
        assert [m["id"] for m in result] == [1, 2]

    def test_importance_tiebreak_when_relevance_equal(self):
        now = datetime.now()
        mems = [
            _mem(1, "episodic", "A", importance=90, last_accessed=now, relevance=0.5),
            _mem(2, "episodic", "B", importance=40, last_accessed=now, relevance=0.5),
        ]
        result = injection._sort_retrieved(mems)
        assert [m["id"] for m in result] == [1, 2]

    def test_empty(self):
        assert injection._sort_retrieved([]) == []


class TestInjectMemories:

    async def test_no_injection_when_query_short(self):
        system_text, injected = await injection.inject_memories(object(), 1, "a")
        assert system_text is None and injected == []

    async def test_three_layers_and_sections(self, monkeypatch):
        prefs = [make_orm_mem(1, "semantic", "用户偏好简洁回复", 80, metadata={"category": "preference", "is_preference": 1})]
        skills = [make_orm_mem(2, "procedural", "先去雾再评估", 70, metadata={"skill": "dehaze", "steps": "..."})]
        retrievals = [_mem(3, "episodic", "上周处理雾图结果满意", 60, relevance=0.8)]
        _install_injection_mocks(monkeypatch, prefs=prefs, skills=skills, es_retrievals=retrievals)

        system_text, injected = await injection.inject_memories(
            object(), 1, "帮我处理雾图", task_type="dehaze"
        )

        assert injected == [
            {
                "memory_id": 1,
                "memory_type": "semantic",
                "content": "用户偏好简洁回复",
                "source": "preference",
            },
            {
                "memory_id": 2,
                "memory_type": "procedural",
                "content": "先去雾再评估",
                "source": "skill",
            },
            {
                "memory_id": 3,
                "memory_type": "episodic",
                "content": "上周处理雾图结果满意",
                "source": "retrieval",
            },
        ]
        assert "【用户画像】" in system_text
        assert "【工作流提示】" in system_text
        assert "【相关记忆】" in system_text

    async def test_scene_trigger_skipped_without_task_type(self, monkeypatch):
        prefs = [make_orm_mem(1, "semantic", "偏好简洁回复", metadata={"is_preference": 1})]
        _install_injection_mocks(monkeypatch, prefs=prefs, user_skills=["dehaze"])

        _, injected = await injection.inject_memories(object(), 1, "帮我处理雾图")
        assert [i["source"] for i in injected] == ["preference"]

    async def test_scene_trigger_fallback_by_keyword(self, monkeypatch):
        prefs = [make_orm_mem(1, "semantic", "偏好简洁回复", metadata={"is_preference": 1})]
        skills = [make_orm_mem(2, "procedural", "先去雾再评估", metadata={"skill": "dehaze"})]
        _install_injection_mocks(
            monkeypatch, prefs=prefs, skills=skills, user_skills=["dehaze"]
        )

        _, injected = await injection.inject_memories(object(), 1, "请执行 dehaze 任务")
        assert [i["source"] for i in injected] == ["preference", "skill"]

    async def test_retrieval_touch_called(self, monkeypatch):
        touch_recorder = []
        retrievals = [_mem(3, "episodic", "历史处理雾图记录", 60, relevance=0.8)]
        _install_injection_mocks(monkeypatch, es_retrievals=retrievals, touch_recorder=touch_recorder)

        await injection.inject_memories(object(), 1, "当前对话", task_type="evaluate")
        assert touch_recorder == [3]

    async def test_retrieval_fallback_to_keyword_when_es_empty(self, monkeypatch):
        touch_recorder = []
        keyword_hits = [make_orm_mem(4, "episodic", "用户反馈：夜间去雾后噪点明显", importance=70)]
        _install_injection_mocks(
            monkeypatch, keyword_hits=keyword_hits, touch_recorder=touch_recorder
        )

        _, injected = await injection.inject_memories(object(), 1, "夜间去雾")
        assert [i["source"] for i in injected] == ["retrieval"]
        assert injected[0]["memory_id"] == 4
        assert touch_recorder == [4]

    async def test_retrieval_sorted_and_truncated_to_limit(self, monkeypatch):
        touch_recorder = []
        retrievals = [
            _mem(1, "episodic", "上周处理雾图，RIDCP 效果最佳", relevance=0.9, importance=90),
            _mem(2, "episodic", "雨天场景用 AOD 去雾偏色严重", relevance=0.8, importance=80),
            _mem(3, "episodic", "用户偏好轻量实时去雾方案", relevance=0.7, importance=70),
            _mem(4, "episodic", "浓雾下先粗去雾再细节增强", relevance=0.6, importance=60),
            _mem(5, "episodic", "车载场景需限制推理耗时", relevance=0.5, importance=50),
            _mem(6, "episodic", "历史对比 DCP 与直方图均衡", relevance=0.4, importance=40),
            _mem(7, "episodic", "夜晚雾图信噪比低，谨慎增强", relevance=0.3, importance=30),
        ]
        _install_injection_mocks(
            monkeypatch, es_retrievals=retrievals, touch_recorder=touch_recorder
        )

        _, injected = await injection.inject_memories(object(), 1, "处理雾图", limit=3)
        retrieval_ids = [i["memory_id"] for i in injected if i["source"] == "retrieval"]
        assert retrieval_ids == [1, 2, 3]
        assert touch_recorder == [1, 2, 3]
