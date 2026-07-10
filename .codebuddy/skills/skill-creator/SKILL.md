---
name: skill-creator
description: Create new skills, modify and improve existing skills, and measure skill performance. Use when users want to create a skill from scratch, update or optimize an existing skill, run evals to test a skill, benchmark skill performance with variance analysis, or optimize a skill's description for better triggering accuracy.
---

# Skill Creator

A skill for creating new skills and iteratively improving them.

## The Core Loop

The process: **Draft the skill → Write test prompts → Run with and without the skill → Evaluate results (qualitative + quantitative) → Improve → Repeat** until satisfied. Then optionally optimize the description for better triggering.

Your job is to figure out where the user is in this loop and help them progress. They might start with a vague idea, a draft, or an existing skill to improve — adapt accordingly.

When communicating, match the user's technical comfort level. Avoid jargon unless the user clearly uses it themselves.

---

## Creating a Skill

### Capture Intent

Understand what the skill should do, when it should trigger, and the expected output format. If the current conversation already contains a workflow the user wants to capture, extract from history first.

### Interview and Research

Ask about edge cases, input/output formats, example files, success criteria, and dependencies. Research via available tools if helpful, to reduce burden on the user.

### Write the SKILL.md

Fill in these components:

- **name**: Skill identifier
- **description**: When to trigger, what it does. This is the primary triggering mechanism — include both what the skill does AND specific contexts for when to use it. Make descriptions slightly "pushy" to combat undertriggering. Example: instead of "How to build a dashboard", write "How to build a dashboard. Use this whenever the user mentions dashboards, data visualization, or wants to display company data."
- **The rest of the skill** — workflows, constraints, examples, etc.

### Skill Writing Guide

#### Anatomy

```
skill-name/
├── SKILL.md (required)
│   ├── YAML frontmatter (name, description required)
│   └── Markdown instructions
└── Bundled Resources (optional)
    ├── scripts/    - Executable code for deterministic/repetitive tasks
    ├── references/ - Docs loaded into context as needed
    └── assets/     - Files used in output (templates, icons, fonts)
```

#### Progressive Disclosure

Skills use three-level loading:
1. **Metadata** (name + description) — Always in context (~100 words)
2. **SKILL.md body** — In context when triggered (ideally <500 lines)
3. **Bundled resources** — As needed (unlimited)

If approaching 500 lines, add a layer of hierarchy with clear pointers to follow-up references. For large reference files (>300 lines), include a table of contents.

When supporting multiple domains, organize by variant:
```
cloud-deploy/
├── SKILL.md (workflow + selection)
└── references/
    ├── aws.md
    ├── gcp.md
    └── azure.md
```

#### Writing Guidance

- Use imperative form in instructions.
- **Explain the why**, not just the what. Today's LLMs understand reasoning — "why this matters" is more effective than "ALWAYS do X". If you find yourself writing all-caps MUSTs, reframe and explain the reasoning instead.
- Include examples, but keep them focused.
- For output formats, provide a clear template.
- Skills must not contain malware, exploits, or misleading content.

#### Defining Output Formats

```markdown
## Report structure
ALWAYS use this exact template:
# [Title]
## Executive summary
## Key findings
## Recommendations
```

### Test Cases

After the skill draft, create 2-3 realistic test prompts and share them with the user. Save to `evals/evals.json` (prompts only — draft assertions later):

```json
{
  "skill_name": "example-skill",
  "evals": [
    {
      "id": 1,
      "prompt": "User's task prompt",
      "expected_output": "Description of expected result",
      "files": []
    }
  ]
}
```

See `references/schemas.md` for the full schema.

---

## Running and Evaluating Test Cases

This is one continuous sequence — don't stop partway through.

Put results in `<skill-name>-workspace/` as a sibling to the skill directory, organized by iteration (`iteration-1/`, `iteration-2/`, ...) and within that, each test case gets a directory.

### Step 1: Spawn all runs in parallel

For each test case, spawn two subagents in the **same turn** — one with the skill, one without (or with the old version for improvements).

**With-skill run:**
```
Execute this task:
- Skill path: <path-to-skill>
- Task: <eval prompt>
- Input files: <eval files if any, or "none">
- Save outputs to: <workspace>/iteration-<N>/eval-<ID>/with_skill/outputs/
```

**Baseline run** — same prompt, different context:
- **New skill**: no skill, save to `without_skill/outputs/`
- **Improving existing skill**: snapshot old version (`cp -r <skill-path> <workspace>/skill-snapshot/`), save to `old_skill/outputs/`

Write `eval_metadata.json` for each test case (assertions empty for now). Use descriptive directory names.

### Step 2: Draft assertions while runs are in progress

Draft quantitative assertions and explain them to the user. Good assertions are objectively verifiable with descriptive names. Update `eval_metadata.json` and `evals/evals.json`.

### Step 3: Capture timing data as runs complete

When each subagent completes, save `total_tokens` and `duration_ms` to `timing.json` in the run directory immediately — this data comes through the notification and isn't persisted elsewhere.

### Step 4: Grade, aggregate, and launch the viewer

1. **Grade each run** — evaluate assertions against outputs, save to `grading.json`. Use `text`, `passed`, `evidence` fields. Prefer scripts over eyeballing when possible.
2. **Aggregate** into benchmark:
   ```bash
   python -m scripts.aggregate_benchmark <workspace>/iteration-N --skill-name <name>
   ```
3. **Analyze** — surface patterns: always-passing assertions (non-discriminating), high-variance evals (possibly flaky), time/token tradeoffs.
4. **Launch the viewer**:
   ```bash
   nohup python <skill-creator-path>/eval-viewer/generate_review.py \
     <workspace>/iteration-N \
     --skill-name "my-skill" \
     --benchmark <workspace>/iteration-N/benchmark.json \
     > /dev/null 2>&1 &
   ```
   For iteration 2+, add `--previous-workspace <workspace>/iteration-<N-1>`.
   In headless environments, use `--static <output_path>` for a standalone HTML file.

5. **Tell the user** the viewer is ready. Two tabs: "Outputs" (qualitative review + feedback) and "Benchmark" (quantitative comparison).

### Step 5: Read the feedback

When the user is done, read `feedback.json`. Empty feedback = the user thought it was fine. Focus improvements on cases with specific complaints. Kill the viewer server when done.

---

## Improving the Skill

After reviewing feedback:

1. **Generalize from feedback** — don't overfit to the few test cases. Branch out with different approaches rather than adding rigid constraints.
2. **Keep the prompt lean** — remove things that aren't pulling their weight. Read transcripts, not just final outputs.
3. **Explain the why** — transmit understanding of the task, not just rote rules. This is more effective than all-caps MUSTs.
4. **Look for repeated work across test cases** — if all runs wrote the same helper script, bundle it in `scripts/`.

Then apply improvements, rerun all test cases into a new iteration, launch the viewer with `--previous-workspace`, and repeat until the user is satisfied, feedback is all empty, or you're not making meaningful progress.

---

## Blind Comparison (Optional)

For rigorous A/B comparison between two skill versions, read `agents/comparator.md` and `agents/analyzer.md`. Give two outputs to an independent agent without telling it which is which, let it judge quality, then analyze the winner. Requires subagents. Most users won't need this.

---

## Description Optimization

The description field in SKILL.md frontmatter determines whether the skill triggers. Offer to optimize it after the skill is complete.

### Step 1: Generate trigger eval queries

Create 20 realistic eval queries — ~10 should-trigger, ~10 should-not-trigger. Save as JSON:

```json
[
  {"query": "the user prompt", "should_trigger": true},
  {"query": "another prompt", "should_trigger": false}
]
```

Queries must be realistic — concrete file paths, column names, company names, personal context. Mix formal and casual phrasing, include edge cases. Negative cases should be near-misses (share keywords but need a different skill), not obviously irrelevant.

### Step 2: Review with user

Use the HTML template from `assets/eval_review.html` to present the eval set. Replace `__EVAL_DATA_PLACEHOLDER__`, `__SKILL_NAME_PLACEHOLDER__`, `__SKILL_DESCRIPTION_PLACEHOLDER__`. Write to a temp file and open it. The user can edit and export the eval set.

### Step 3: Run the optimization loop

```bash
python -m scripts.run_loop \
  --eval-set <path-to-trigger-eval.json> \
  --skill-path <path-to-skill> \
  --model <model-id-powering-this-session> \
  --max-iterations 5 \
  --verbose
```

The script splits 60/40 train/test, evaluates each description with 3 runs per query, and iterates with extended thinking to improve. It selects `best_description` by test score to avoid overfitting.

**Note:** Eval queries should be substantive enough that a skill would actually be consulted. Simple one-step queries like "read file X" won't trigger skills regardless of description quality.

### Step 4: Apply the result

Update the skill's SKILL.md frontmatter with `best_description`. Show before/after and report scores.

---

## Package and Present

If the `present_files` tool is available, package the skill:

```bash
python -m scripts.package_skill <path/to/skill-folder>
```

---

## Platform-Specific Behavior

| Feature | Claude Code / Cowork | Claude.ai |
|---------|---------------------|-----------|
| Subagents (parallel runs) | ✅ Use full workflow | ❌ Run test cases yourself, one at a time |
| Baseline comparison | ✅ Spawn without-skill/old-skill subagents | ❌ Skip baseline |
| Quantitative benchmarking | ✅ Full grading + benchmark.json | ❌ Skip, rely on qualitative feedback |
| Browser viewer | ✅ Use `generate_review.py` | ❌ Present results inline |
| Description optimization | ✅ `run_loop.py` | ❌ Requires `claude -p` CLI |
| Blind comparison | ✅ Requires subagents | ❌ Skip |
| Packaging | ✅ `package_skill.py` | ✅ `package_skill.py` |
| Headless (no display) | ✅ Use `--static` flag | N/A |

---

## Reference Files

- `agents/grader.md` — How to evaluate assertions against outputs
- `agents/comparator.md` — How to do blind A/B comparison
- `agents/analyzer.md` — How to analyze why one version beat another
- `references/schemas.md` — JSON structures for evals.json, grading.json, etc.
