package com.pei.clarity.common.util;

import java.util.ArrayList;
import java.util.List;

/**
 * 跨端共享 Mongo 集合的 BSON 数值编码工具（对齐 clarity-python {@code app/repository/mongo_bson.py}）。
 *
 * <p>{@code login_log} / {@code audit_log} / {@code ai_api_call_log} 三个集合由 java、go、python
 * 共写共读：java/go 写入的整数落 BSON <b>int64</b>，python(pymongo) 对绝对值 &lt; 2^31 的整数落
 * <b>int32</b>，而 BSON 数值类型不相等 ⇒ <b>单类型条件会静默漏读</b>对端写入与改造前的历史记录
 * （返回空列表、不报错）。故跨端共享集合上的整数归属字段（{@code user_id} / {@code key_id} /
 * {@code operator_id} / {@code target_id}）一律用本工具给出的 int32/int64 双类型
 * {@code $in} 候选查询，禁止用 {@code is(...)} 单类型匹配。
 */
public final class MongoBsonUtils {

    /** int32 值域（Integer 承载 int32，Long 承载 int64） */
    private static final long INT32_MIN = Integer.MIN_VALUE;

    private static final long INT32_MAX = Integer.MAX_VALUE;

    private MongoBsonUtils() {}

    /**
     * 整数归属字段的 BSON 双类型候选（入参不可为 {@code null}，条件缺失应由调用方先行判空）。
     *
     * <p>{@code Long} 由驱动落 int64（java/go 口径）、{@code Integer} 落 int32
     * （python 历史记录与 pymongo 的小整数口径）；超出 int32 值域时只保留 int64
     * ——python 对 ≥ 2^31 的整数同样编码为 int64（且 {@code intValue()} 会溢出）。
     */
    public static List<Object> int32Int64Candidates(Long value) {
        List<Object> candidates = new ArrayList<>(2);
        candidates.add(value);
        if (value >= INT32_MIN && value <= INT32_MAX) {
            candidates.add(value.intValue());
        }
        return candidates;
    }
}
