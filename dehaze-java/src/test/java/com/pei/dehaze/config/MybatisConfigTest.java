package com.pei.dehaze.config;

import com.baomidou.mybatisplus.extension.plugins.MybatisPlusInterceptor;
import com.baomidou.mybatisplus.extension.plugins.inner.PaginationInnerInterceptor;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * 分页兜底配置测试。
 * <p>
 * 参数校验（{@code @Max(100)}+{@code @Valid}）负责报 A0400，本拦截器的 {@code maxLimit}
 * 是校验漏网时的第二道防线：无上限则单次请求可直接拉出任意大结果集拖垮 DB。
 */
@DisplayName("MybatisConfig 分页兜底测试")
class MybatisConfigTest {

    @Test
    @DisplayName("分页插件必须配置单页上限兜底")
    void paginationInterceptor_hasMaxLimitFallback() {
        MybatisPlusInterceptor interceptor = new MybatisConfig().mybatisPlusInterceptor();

        PaginationInnerInterceptor pagination = interceptor.getInterceptors().stream()
                .filter(PaginationInnerInterceptor.class::isInstance)
                .map(PaginationInnerInterceptor.class::cast)
                .findFirst()
                .orElseThrow(() -> new AssertionError("未注册分页插件"));

        assertThat(pagination.getMaxLimit()).isEqualTo(MybatisConfig.PAGE_SIZE_MAX_LIMIT);
    }
}
