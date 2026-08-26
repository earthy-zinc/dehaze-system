from app.service.ai.paradigms import reflexion


def _fake_model(response: str):
    async def _call(messages, system_prompt):
        return response

    return _call


async def test_evaluate_output_parses_score_and_feedback():
    score, feedback = await reflexion.evaluate_output(
        "生成报告", "报告内容", _fake_model('{"score": 0.9, "feedback": "符合要求"}')
    )
    assert score == 0.9
    assert "符合要求" in feedback


async def test_evaluate_output_clamps_score():
    score, _ = await reflexion.evaluate_output("任务", "输出", _fake_model('{"score": 5}'))
    assert score == 1.0


async def test_evaluate_output_expected_format_penalizes():
    score, feedback = await reflexion.evaluate_output(
        "输出JSON",
        "plain text not json",
        _fake_model('{"score": 1.0, "feedback": "ok"}'),
        expected="json",
    )
    assert score == 0.5
    assert "json" in feedback.lower()


async def test_evaluate_output_parse_failure_low_score():
    score, _ = await reflexion.evaluate_output("任务", "输出", _fake_model("不是JSON"))
    assert score == 0.0


async def test_reflexion_loop_accepts_when_above_threshold():
    calls = {"n": 0}

    async def actor(messages, prompt):
        calls["n"] += 1
        return "第一次输出"

    async def evaluate(requirement, output):
        return (0.95, "好")

    async def reflect(requirement, output, feedback):
        return {"root_cause": "x", "strategy": "y"}

    loop = reflexion.reflexion_loop(
        run_actor=actor, evaluate=evaluate, reflect=reflect, max_iterations=5, threshold=0.8
    )
    best, rounds = await loop("任务", [])
    assert best == "第一次输出"
    assert calls["n"] == 1
    assert len(rounds) == 1


async def test_reflexion_loop_returns_best_after_max_iterations():
    outputs = iter(["低分1", "低分2", "好输出"])

    async def actor(messages, prompt):
        return next(outputs)

    scores = iter([0.3, 0.4, 0.85])

    async def evaluate(requirement, output):
        s = next(scores)
        return (s, f"分{s}")

    async def reflect(requirement, output, feedback):
        return {"root_cause": "r", "strategy": "改进"}

    loop = reflexion.reflexion_loop(
        run_actor=actor, evaluate=evaluate, reflect=reflect, max_iterations=3, threshold=0.8
    )
    best, rounds = await loop("任务", [])
    assert best == "好输出"
    assert len(rounds) == 3


async def test_reflexion_loop_never_reaches_threshold():
    outputs = iter(["输出A", "输出B"])

    async def actor(messages, prompt):
        return next(outputs)

    async def evaluate(requirement, output):
        return (0.6 if output == "输出A" else 0.7, "均不达标")

    async def reflect(requirement, output, feedback):
        return {"root_cause": "r", "strategy": f"改-{output}"}

    loop = reflexion.reflexion_loop(
        run_actor=actor, evaluate=evaluate, reflect=reflect, max_iterations=2, threshold=0.8
    )
    best, rounds = await loop("任务", [])
    assert best == "输出B"
    assert len(rounds) == 2


async def test_reflexion_loop_injects_reflection_strategy():
    seen_prompts = []

    async def actor(messages, prompt):
        seen_prompts.append(prompt)
        return "输出"

    async def evaluate(requirement, output):
        return (0.5, "待改进")

    async def reflect(requirement, output, feedback):
        return {"root_cause": "r", "strategy": "避免重复错误"}

    loop = reflexion.reflexion_loop(
        run_actor=actor, evaluate=evaluate, reflect=reflect, max_iterations=2, threshold=0.8
    )
    await loop("任务", [])
    assert "避免重复错误" in seen_prompts[1]


def test_build_reflection_memory_source_and_skill():
    memory = reflexion.build_reflection_memory(
        user_id=10,
        conversation_id=20,
        model_id="gpt-4o",
        requirement="去雾任务",
        reflection={"root_cause": "边缘失真", "strategy": "改用引导滤波"},
        skill="dehaze",
    )
    assert memory["source"] == "reflection"
    assert memory["source_type"] == "self_reflection"
    assert memory["metadata"] == {"skill": "dehaze"}
    assert "边缘失真" in memory["content"]
    assert "改用引导滤波" in memory["content"]
