package com.pei.dehaze.common.enums;

import com.baomidou.mybatisplus.annotation.EnumValue;
import com.fasterxml.jackson.annotation.JsonValue;
import com.pei.dehaze.common.base.IBaseEnum;
import lombok.Getter;

import java.util.Set;

/**
 * 算法生命周期状态枚举
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
@Getter
public enum AlgorithmStatusEnum implements IBaseEnum<Integer> {

    DRAFT(0, "草稿"),
    TESTING(1, "测试中"),
    PENDING_REVIEW(2, "待审核"),
    PUBLISHED(3, "已发布"),
    DISABLED(4, "已停用"),
    ARCHIVED(5, "已归档");

    @JsonValue
    @EnumValue
    private final Integer value;

    private final String label;

    AlgorithmStatusEnum(Integer value, String label) {
        this.value = value;
        this.label = label;
    }

    /**
     * 可编辑的状态（草稿、测试中）
     */
    public static final Set<Integer> EDITABLE_STATUSES = Set.of(DRAFT.value, TESTING.value);

    /**
     * 可删除的状态（草稿、已停用、已归档）
     */
    public static final Set<Integer> DELETABLE_STATUSES = Set.of(DRAFT.value, DISABLED.value, ARCHIVED.value);

    /**
     * 终态（已发布、已停用、已归档）
     */
    public static final Set<Integer> FINAL_STATUSES = Set.of(PUBLISHED.value, DISABLED.value, ARCHIVED.value);
}
