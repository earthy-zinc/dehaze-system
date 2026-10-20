package com.pei.clarity.service.impl;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

/**
 * 系统预置字典类型编码集合（T-DM-025）三端一致性护栏。
 *
 * <p>事实源：python {@code app/service/dict_service.py} 的 {@code SYSTEM_PRESET_DICT_TYPE_CODES}。
 * 该集合决定"系统预置字典类型不可删除"的保护范围；任一端少一项，就会允许删除事实源禁止删除的
 * 字典类型（曾出现 java/go 缺 {@code ai_model}，于是可删掉"平台默认模型"字典类型而 python 禁止，
 * 造成三端行为分裂）。本用例与 go 端 {@code sys_dict_type_preset_test.go} 对称，钉住同一集合。
 */
@DisplayName("SysDictTypeServiceImpl 预置字典类型码集合")
class SysDictTypePresetCodesTest {

    /** 期望集合 == python 事实源集合（顺序无关）。 */
    private static final Set<String> EXPECTED_PRESET_CODES = Set.of(
            "gender",
            "ai_guardrail_defaults",
            "ai_provider_health",
            "ai_embedding",
            "ai_model",
            "member_growth_rules",
            "favorite_capacity",
            "ai_eval");

    @SuppressWarnings("unchecked")
    private static Set<String> actualPresetCodes() {
        List<String> codes = (List<String>)
                ReflectionTestUtils.getField(SysDictTypeServiceImpl.class, "SYSTEM_PRESET_DICT_TYPE_CODES");
        assertThat(codes).as("SYSTEM_PRESET_DICT_TYPE_CODES 常量必须存在").isNotNull();
        return Set.copyOf(codes);
    }

    @Test
    @DisplayName("预置码集合与 python 事实源逐项相等（顺序无关）；漂移即会让 java 允许删除 python 保护的字典类型")
    void presetCodesMatchPythonFactSource() {
        assertThat(actualPresetCodes())
                .as("预置字典类型码集合必须与 python SYSTEM_PRESET_DICT_TYPE_CODES 完全一致")
                .isEqualTo(EXPECTED_PRESET_CODES);
    }
}
