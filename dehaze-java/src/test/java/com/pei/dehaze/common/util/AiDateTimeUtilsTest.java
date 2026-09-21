package com.pei.dehaze.common.util;

import com.pei.dehaze.common.exception.BusinessException;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.time.LocalDateTime;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * AiDateTimeUtils 单元测试：锁定与 python `_parse_datetime` 一致的三种入参格式
 * （空格分隔 / ISO / 仅日期按当日 00:00:00）与严格/宽松两种失败策略。
 */
@DisplayName("AiDateTimeUtils 单元测试")
class AiDateTimeUtilsTest {

    @Test
    @DisplayName("空格分隔、ISO、仅日期三种格式均可解析，仅日期落到当日 00:00:00")
    void parsesSupportedFormats() {
        assertThat(AiDateTimeUtils.parseOrNull("2026-09-17 10:30:00"))
                .isEqualTo(LocalDateTime.of(2026, 9, 17, 10, 30, 0));
        assertThat(AiDateTimeUtils.parseOrNull("2026-09-17T10:30:00"))
                .isEqualTo(LocalDateTime.of(2026, 9, 17, 10, 30, 0));
        assertThat(AiDateTimeUtils.parseOrNull("2026-09-17"))
                .isEqualTo(LocalDateTime.of(2026, 9, 17, 0, 0, 0));
    }

    @Test
    @DisplayName("空值与非法格式：宽松解析返回 null（非法按无过滤），严格解析抛 A0400")
    void distinguishesLenientAndStrictFailure() {
        assertThat(AiDateTimeUtils.parseOrNull(null)).isNull();
        assertThat(AiDateTimeUtils.parseOrNull("  ")).isNull();
        assertThat(AiDateTimeUtils.parseOrNull("2026/09/17")).isNull();
        // 严格解析：空值 = 无过滤条件，非法值 = A0400
        assertThat(AiDateTimeUtils.parse(null)).isNull();
        assertThat(AiDateTimeUtils.parse("  ")).isNull();
        assertThatThrownBy(() -> AiDateTimeUtils.parse("2026/09/17"))
                .isInstanceOf(BusinessException.class);
    }
}
