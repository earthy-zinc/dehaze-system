import math
import random

import tiktoken

from app.service.kb.chunking_engine import _count_tokens, chunk_text


class TestFixed:
    def test_chunk_count_ceil_division(self):
        text = "由于分块算法需要兼顾语义完整性与长度约束，系统优先在句末标点处切割；" * 30
        tokens = _count_tokens(text)
        step = 500 - 50
        chunks = chunk_text(text, "fixed", chunk_size=500, chunk_overlap=50)
        assert math.ceil(tokens / step) - 1 <= len(chunks) <= math.ceil(tokens / step)

    def test_each_chunk_within_limit(self):
        text = "去雾系统利用大气散射模型对图像进行退化复原，并保留细节与色彩。" * 25
        chunks = chunk_text(text, "fixed", chunk_size=300, chunk_overlap=30)
        assert all(c.token_count <= 300 for c in chunks)

    def test_overlap_between_adjacent(self):
        text = "The image dehazing system estimates transmission maps and recovers clear images. " * 40
        chunks = chunk_text(text, "fixed", chunk_size=120, chunk_overlap=40)
        assert len(chunks) >= 2
        enc = tiktoken.get_encoding("cl100k_base")
        overlap_text = enc.decode(enc.encode(chunks[0].content)[-40:])
        assert chunks[1].content.startswith(overlap_text.strip())

    def test_no_content_loss_with_overlap(self):
        sentences = ["这是第一句话。" * 20, "这是第二句话。" * 20, "这是第三句话。" * 20]
        text = "\n".join(sentences)
        chunks = chunk_text(text, "fixed", chunk_size=150, chunk_overlap=20)
        assert all(c.content for c in chunks)
        assert text[:20] in chunks[0].content
        assert chunks[-1].content.rstrip().endswith("。")


class TestSemantic:
    def test_split_at_paragraph_boundary(self):
        p1 = "第一段内容。" * 60
        p2 = "第二段内容。" * 60
        text = f"{p1}\n\n{p2}"
        chunks = chunk_text(text, "semantic", chunk_size=500, chunk_overlap=0)
        assert len(chunks) == 2
        assert chunks[0].content.strip() == p1.strip()
        assert chunks[1].content.strip() == p2.strip()

    def test_long_paragraph_split_at_sentence(self):
        para = "这是一个很长的段落。" * 120
        chunks = chunk_text(para, "semantic", chunk_size=200, chunk_overlap=0)
        assert len(chunks) > 1
        for c in chunks[:-1]:
            assert c.content.rstrip().endswith(("。", "？", "！", "?", "!"))

    def test_short_paragraph_merged_with_next(self):
        s1 = "短一。"
        s2 = "短二。"
        s3 = "短三。"
        text = f"{s1}\n\n{s2}\n\n{s3}"
        chunks = chunk_text(text, "semantic", chunk_size=800, chunk_overlap=0)
        assert len(chunks) == 1
        assert "短一" in chunks[0].content and "短三" in chunks[0].content


class TestRecursive:
    def test_split_priority_newline_then_sentence(self):
        paras = ["段落一的第一句。段落一的第二句。" * 5, "段落二的第一句。段落二的第二句。" * 5]
        text = "\n\n".join(paras)
        chunks = chunk_text(text, "recursive", chunk_size=120, chunk_overlap=0)
        assert len(chunks) >= 2
        for c in chunks:
            assert "段落一" not in c.content or "段落二" not in c.content

    def test_respect_comma_and_period_order(self):
        clause = "这是一大段没有换行的文字，包含很多逗号，用来测试递归切分的优先级，"
        text = clause * 15 + "结束。"
        chunks = chunk_text(text, "recursive", chunk_size=80, chunk_overlap=0)
        assert len(chunks) > 1
        for c in chunks[:-1]:
            assert c.token_count <= 80


class TestQa:
    def test_q_a_split_into_separate_chunks(self):
        text = "Q: 什么是去雾？\nA: 去雾是一种图像增强技术。\nQ: 如何使用？\nA: 上传图片即可。"
        chunks = chunk_text(text, "qa", chunk_size=800, chunk_overlap=0)
        types = [c.metadata["type"] for c in chunks]
        assert types.count("question") == 2
        assert types.count("answer") == 2

    def test_metadata_type_question_answer(self):
        text = "Q: 问题内容\nA: 答案内容"
        chunks = chunk_text(text, "qa", chunk_size=800, chunk_overlap=0)
        assert chunks[0].metadata["type"] == "question"
        assert chunks[1].metadata["type"] == "answer"

    def test_chunk_index_adjacent(self):
        text = "Q: 一\nA: 一答\nQ: 二\nA: 二答"
        chunks = chunk_text(text, "qa", chunk_size=800, chunk_overlap=0)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))


class TestTable:
    def test_table_kept_whole(self):
        table = "\n".join([f"| 列{i} | 值{i} |" for i in range(10)])
        text = f"前言\n\n{table}\n\n后记"
        chunks = chunk_text(text, "table", chunk_size=800, chunk_overlap=0)
        table_chunk = next(c for c in chunks if c.metadata.get("type") == "table")
        assert "列0" in table_chunk.content and "值9" in table_chunk.content

    def test_metadata_rows_cols(self):
        table = (
            "| 名称 | 数量 | 单价 |\n"
            "| --- | --- | --- |\n"
            "| 苹果 | 5 | 2 |\n"
            "| 香蕉 | 3 | 4 |\n"
            "| 橘子 | 8 | 3 |"
        )
        chunks = chunk_text(table, "table", chunk_size=800, chunk_overlap=0)
        t = next(c for c in chunks if c.metadata.get("type") == "table")
        assert t.metadata["rows"] == 3
        assert t.metadata["cols"] == 3


class TestPostprocess:
    def test_merge_short_chunks(self):
        text = "短" * 10 + "。" + "长文本" * 100 + "。"
        chunks = chunk_text(text, "fixed", chunk_size=200, chunk_overlap=0)
        for c in chunks[:-1]:
            assert c.token_count >= 50

    def test_force_split_over_2000(self):
        text = "图像去雾是计算机视觉的重要分支，它基于物理模型估计传输率。" * 70
        chunks = chunk_text(text, "fixed", chunk_size=2000, chunk_overlap=0)
        assert len(chunks) > 1
        assert all(c.token_count <= 2000 for c in chunks)

    def test_whitespace_cleaned(self):
        text = "   多余空白\t\t换行\n\n\n\n压缩  "
        chunks = chunk_text(text, "fixed", chunk_size=800, chunk_overlap=0)
        assert chunks
        assert "  " not in chunks[0].content
        assert "\n\n\n" not in chunks[0].content

    def test_chunk_index_attached(self):
        text = "测试文本。" * 50
        chunks = chunk_text(text, "fixed", chunk_size=100, chunk_overlap=0)
        assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


class TestEdgeCases:
    def test_empty_text_returns_empty(self):
        assert chunk_text("", "fixed", 800, 80) == []

    def test_whitespace_only_returns_empty(self):
        assert chunk_text("   \n\n\t  ", "fixed", 800, 80) == []

    def test_chunk_size_clamped_min(self):
        text = "测试文本内容。" * 80
        small = chunk_text(text, "fixed", chunk_size=10, chunk_overlap=0)
        clamped = chunk_text(text, "fixed", chunk_size=50, chunk_overlap=0)
        assert len(small) == len(clamped)
        assert [c.token_count for c in small] == [c.token_count for c in clamped]


def _random_dirty_text(rng, min_len=200, max_len=4000):
    en_words = ["the", "system", "data", "error", "config", "server", "用户", "处理"]
    zh_chars = "这是去雾图像增强系统的核心模块用于处理各种复杂场景下的文档解析与分块逻辑"
    full_punc = "，。！？：；（）【】"
    half_punc = ",.!??:;()[]"
    pool = (
        [en_words[rng.randint(0, len(en_words) - 1)] + " " for _ in range(3)]
        + [zh_chars[rng.randint(0, len(zh_chars) - 1)] for _ in range(3)]
        + [rng.choice(full_punc + half_punc) for _ in range(2)]
        + ["\n" * rng.randint(1, 3) for _ in range(rng.randint(1, 3))]
    )
    text = "".join(pool)
    while len(text) < min_len:
        text += "".join(rng.choice(pool) for _ in range(3))
    return text[:max_len]


class TestAdversarialDirtyCorpus:
    def test_fullwidth_halfwidth_mixed_punctuation(self):
        text = "第一句，second sentence. 第三句：config：value（ok）\n第四句, end."
        chunks = chunk_text(text, "fixed", chunk_size=800, chunk_overlap=0)
        assert chunks and all(c.content for c in chunks)

    def test_crlf_lf_mixed_and_leading_trailing_spaces(self):
        text = "  行一\r\n  行二\n行三\r\n  行四  \n\n\r\n行五"
        chunks = chunk_text(text, "fixed", chunk_size=800, chunk_overlap=0)
        assert chunks
        joined = "\n".join(c.content for c in chunks)
        assert "行一" in joined and "行五" in joined

    def test_consecutive_blank_lines_and_tabs(self):
        text = "段落甲\n\n\n\n段落乙\t\t段落丙\n\n\n段落丁"
        chunks = chunk_text(text, "fixed", chunk_size=800, chunk_overlap=0)
        joined = "\n".join(c.content for c in chunks)
        assert "\n\n\n" not in joined
        assert "段落甲" in joined and "段落丁" in joined
        assert "段落乙 段落丙" in joined

    def test_bom_and_zero_width_space_and_emoji(self):
        text = "\ufeff\u200b系统启动🚀完成✅" + "正常内容。" * 40
        chunks = chunk_text(text, "fixed", chunk_size=800, chunk_overlap=0)
        assert chunks
        joined = "".join(c.content for c in chunks)
        assert "系统启动" in joined and "正常内容" in joined

    def test_long_unbroken_line_no_separator(self):
        line = "abcdefghABCDEFGH0123456789" * 400
        chunks = chunk_text(line, "recursive", chunk_size=500, chunk_overlap=0)
        assert chunks
        assert all(c.content for c in chunks)

    def test_chinese_english_digit_mixed_dense_paragraph(self):
        text = ("部署 Deployment 阶段需要配置 config.yaml，共 3 个步骤：" * 30) + "\n"
        text += ("Step 1: 初始化 init()，耗时约 12s；Step 2: 加载 load()，耗时 8s。" * 30)
        chunks = chunk_text(text, "semantic", chunk_size=600, chunk_overlap=0)
        assert chunks and all(c.content for c in chunks)


class TestTableVariantRecognition:
    def test_code_block_pipe_not_misidentified_as_table(self):
        md = "说明\n```python\n| a | b |\n| --- | --- |\n| 1 | 2 |\n```\n结尾"
        chunks = chunk_text(md, "table", chunk_size=800, chunk_overlap=0)
        table_chunks = [c for c in chunks if c.metadata.get("type") == "table"]
        assert len(table_chunks) == 0

    def test_single_cell_table(self):
        table = "| 项目A |\n| --- |\n| 值1 |\n| 值2 |"
        chunks = chunk_text(table, "table", chunk_size=800, chunk_overlap=0)
        t = next(c for c in chunks if c.metadata.get("type") == "table")
        assert t.metadata["cols"] == 1
        assert t.metadata["rows"] == 2

    def test_separator_without_leading_trailing_pipe(self):
        table = "名称 | 数量\n---|---\n苹果 | 5\n香蕉 | 3"
        chunks = chunk_text(table, "table", chunk_size=800, chunk_overlap=0)
        assert all(c.metadata.get("type") != "table" for c in chunks)
        assert any("苹果" in c.content for c in chunks)

    def test_cell_content_with_newline(self):
        table = "| 名称 | 描述 |\n| --- | --- |\n| 苹果 | 好吃\n很甜 |"
        chunks = chunk_text(table, "table", chunk_size=800, chunk_overlap=0)
        joined = "\n".join(c.content for c in chunks)
        assert "好吃" in joined and "很甜" in joined


class TestQAVariantRecognition:
    def test_fullwidth_colon(self):
        text = "Q：什么是去雾？\nA：一种图像增强技术。"
        chunks = chunk_text(text, "qa", chunk_size=800, chunk_overlap=0)
        assert [c.metadata["type"] for c in chunks] == ["question", "answer"]

    def test_lowercase_q_a(self):
        chunks = chunk_text("q: hello\na: world", "qa", chunk_size=800, chunk_overlap=0)
        assert [c.metadata["type"] for c in chunks] == ["question", "answer"]

    def test_a_without_q(self):
        text = "A：这是独立的答案内容。"
        chunks = chunk_text(text, "qa", chunk_size=800, chunk_overlap=0)
        assert chunks[0].metadata["type"] == "answer"

    def test_consecutive_qa_pairs(self):
        text = "Q: 一\nA: 一答\nQ: 二\nA: 二答\nQ: 三\nA: 三答"
        chunks = chunk_text(text, "qa", chunk_size=800, chunk_overlap=0)
        assert len(chunks) == 6
        assert all(c.metadata["chunk_index"] == i for i, c in enumerate(chunks))

    def test_qa_content_multiline(self):
        text = "Q: 如何操作\n第一步\n第二步\nA: 按上述步骤执行即可"
        chunks = chunk_text(text, "qa", chunk_size=800, chunk_overlap=0)
        types = [c.metadata["type"] for c in chunks]
        assert types == ["question", "answer"]

    def test_qa_no_space_after_colon(self):
        text = "Q:问题内容\nA:答案内容"
        chunks = chunk_text(text, "qa", chunk_size=800, chunk_overlap=0)
        assert [c.metadata["type"] for c in chunks] == ["question", "answer"]


class TestRecursiveVariant:
    def test_only_separators_input(self):
        assert chunk_text("\n\n\n\n", "recursive", chunk_size=100, chunk_overlap=0) == []
        dots = chunk_text("。。。。", "recursive", chunk_size=100, chunk_overlap=0)
        assert dots and dots[0].content.strip() == "。。。。"


class TestParameterBoundaries:
    def test_chunk_size_min_value_50(self):
        text = "测试。" * 100
        chunks = chunk_text(text, "fixed", chunk_size=50, chunk_overlap=0)
        assert chunks and all(c.content for c in chunks)

    def test_overlap_equals_chunk_size_minus_one(self):
        text = "测" * 200
        chunks = chunk_text(text, "fixed", chunk_size=50, chunk_overlap=49)
        assert len(chunks) >= 2

    def test_single_char_text(self):
        chunks = chunk_text("好", "fixed", 800, 0)
        assert len(chunks) == 1 and chunks[0].content == "好" and chunks[0].token_count == 1

    def test_whitespace_only_text(self):
        assert chunk_text("   \n\t  \n\n ", "fixed", 800, 0) == []


class TestInvariantRandomCorpus:
    STRATEGIES = ("fixed", "semantic", "recursive", "qa", "table")

    def test_random_dirty_texts_invariants(self):
        rng = random.Random(20260821)
        for _ in range(50):
            text = _random_dirty_text(rng)
            for strategy in self.STRATEGIES:
                chunks = chunk_text(text, strategy, chunk_size=500, chunk_overlap=50)
                assert isinstance(chunks, list)
                for c in chunks:
                    assert c.content, f"{strategy} 产生空块: {text!r}"
                    assert c.token_count >= 0
                    assert isinstance(c.metadata, dict)
                indices = [c.metadata.get("chunk_index") for c in chunks]
                assert indices == list(range(len(chunks)))
                if strategy == "fixed":
                    assert all(c.token_count <= 500 for c in chunks)
                if strategy in ("semantic", "recursive"):
                    for c in chunks:
                        assert "\n\n\n" not in c.content
