import pytest

from app.service.ai.paradigms import plan_execute


def _fake_model(response: str):
    async def _call(messages, system_prompt):
        return response

    return _call


async def test_build_plan_parses_structured_json():
    raw = (
        '{"tasks": ['
        '{"id": "A", "description": "分析", "depends_on": [], '
        '"tool_hint": "analyze", "paradigm": "react"},'
        '{"id": "B", "description": "处理", "depends_on": ["A"], "tool_hint": null},'
        '{"id": "C", "description": "复核", "depends_on": ["B"], "paradigm": "reflexion"}]}'
    )
    plan = await plan_execute.build_plan("批量处理任务", _fake_model(raw))
    assert plan["status"] == "pending"
    assert len(plan["tasks"]) == 3
    by_id = {t["id"]: t for t in plan["tasks"]}
    assert by_id["A"]["depends_on"] == []
    assert by_id["B"]["depends_on"] == ["A"]
    assert by_id["C"]["paradigm"] == "reflexion"
    assert by_id["C"]["depends_on"] == ["B"]
    assert by_id["A"]["tool_hint"] == "analyze"


async def test_build_plan_falls_back_on_parse_error():
    plan = await plan_execute.build_plan("任务", _fake_model("不是JSON"))
    assert len(plan["tasks"]) == 1
    assert plan["tasks"][0]["id"] == "A"
    assert plan["tasks"][0]["paradigm"] == "react"


def _tasks(spec):
    return [
        {"id": tid, "description": tid, "depends_on": deps, "paradigm": "react"}
        for tid, deps in spec.items()
    ]


def test_compute_batches_parallel_then_serial():
    tasks = _tasks(
        {
            "A": [],
            "B": ["A"],
            "C": [],
            "D": [],
            "E": ["B", "D"],
        }
    )
    batches = plan_execute.compute_batches(tasks)
    assert sorted(batches[0]) == ["A", "C", "D"]
    assert batches[1] == ["B"]
    assert batches[2] == ["E"]
    assert sorted(tid for b in batches for tid in b) == sorted(["A", "B", "C", "D", "E"])


def test_compute_batches_single_batch_independent():
    tasks = _tasks({"A": [], "B": [], "C": []})
    batches = plan_execute.compute_batches(tasks)
    assert len(batches) == 1
    assert sorted(batches[0]) == ["A", "B", "C"]


def test_compute_batches_chain_serial():
    tasks = _tasks({"A": [], "B": ["A"], "C": ["B"]})
    batches = plan_execute.compute_batches(tasks)
    assert batches == [["A"], ["B"], ["C"]]


def test_compute_batches_cycle_falls_back():
    tasks = _tasks({"A": ["B"], "B": ["A"]})
    batches = plan_execute.compute_batches(tasks)
    assert sorted(tid for b in batches for tid in b) == ["A", "B"]


async def test_replan_records_revision_and_keeps_unaffected():
    plan = plan_execute.new_plan()
    plan["tasks"] = [
        {
            "id": "A",
            "description": "a",
            "depends_on": [],
            "paradigm": "react",
            "status": "done",
            "result": "ok",
        },
        {
            "id": "B",
            "description": "b",
            "depends_on": [],
            "paradigm": "react",
            "status": "failed",
            "result": "",
        },
        {
            "id": "C",
            "description": "c",
            "depends_on": [],
            "paradigm": "react",
            "status": "done",
            "result": "ok",
        },
    ]
    raw = (
        '{"revised": [{"id": "B2", "description": "b-修订", "depends_on": [], "tool_hint": null}]}'
    )
    await plan_execute.replan(plan, ["B"], _fake_model(raw))

    assert plan["status"] == "revised"
    assert len(plan["tasks"]) == 3
    assert "A" in [t["id"] for t in plan["tasks"]]
    assert "C" in [t["id"] for t in plan["tasks"]]
    assert "B2" in [t["id"] for t in plan["tasks"]]
    assert "B" not in [t["id"] for t in plan["tasks"]]
    assert len(plan["revisions"]) == 1
    assert plan["revisions"][0]["reason"] == "B"


async def test_replan_no_failed_is_noop():
    plan = plan_execute.new_plan()
    plan["tasks"] = [
        {
            "id": "A",
            "description": "a",
            "depends_on": [],
            "paradigm": "react",
            "status": "done",
            "result": "ok",
        }
    ]
    await plan_execute.replan(plan, [], _fake_model("{}"))
    assert plan["status"] == "pending"


def _plan_with_status(status):
    plan = plan_execute.new_plan()
    plan["status"] = status
    plan["tasks"] = [
        {
            "id": "A",
            "description": "a",
            "depends_on": [],
            "paradigm": "react",
            "status": "pending",
            "result": None,
        },
        {
            "id": "B",
            "description": "b",
            "depends_on": [],
            "paradigm": "react",
            "status": "pending",
            "result": None,
        },
        {
            "id": "C",
            "description": "c",
            "depends_on": [],
            "paradigm": "react",
            "status": "pending",
            "result": None,
        },
    ]
    return plan


def test_plan_edit_remove_reorder_add():
    plan = _plan_with_status("pending")
    plan_execute.apply_plan_edit(
        plan,
        {
            "remove": ["B"],
            "reorder": ["C", "A"],
            "add": {"description": "新增任务", "depends_on": ["A"]},
        },
    )
    ids = [t["id"] for t in plan["tasks"]]
    assert ids[0] == "C"
    assert ids[1] == "A"
    assert "B" not in ids
    assert len(plan["tasks"]) == 3


def test_plan_edit_add_cleans_removed_depends():
    plan = _plan_with_status("pending")
    plan["tasks"].append(
        {
            "id": "D",
            "description": "d",
            "depends_on": ["B"],
            "paradigm": "react",
            "status": "pending",
            "result": None,
        }
    )
    plan_execute.apply_plan_edit(plan, {"remove": ["B"]})
    d = next(t for t in plan["tasks"] if t["id"] == "D")
    assert d["depends_on"] == []


def test_plan_edit_rejected_when_executing():
    plan = _plan_with_status("executing")
    with pytest.raises(ValueError):
        plan_execute.apply_plan_edit(plan, {"remove": ["A"]})


def test_plan_edit_none_is_noop():
    plan = _plan_with_status("pending")
    assert plan_execute.apply_plan_edit(plan, None) is plan
