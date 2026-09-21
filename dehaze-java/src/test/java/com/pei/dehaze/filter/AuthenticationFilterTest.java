package com.pei.dehaze.filter;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.springframework.mock.web.MockHttpServletRequest;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * 认证过滤器豁免清单单元测试。
 * <p>
 * 第三方协议端点的凭据形态（OpenAI Bearer / Claude x-api-key）本过滤器不识别，必须放行到 python 终审；
 * 其余受保护路径不得因豁免清单被误放行。
 */
@DisplayName("AuthenticationFilter 鉴权豁免单元测试")
class AuthenticationFilterTest {

    private final AuthenticationFilter filter = new AuthenticationFilter(null, null);

    @ParameterizedTest(name = "{0} {1} 豁免")
    @DisplayName("第三方协议端点与 A2A 全局入口豁免本地鉴权")
    @CsvSource({
            "POST, /api/v1/chat/completions",
            "POST, /api/v1/messages",
            "GET,  /api/v1/models",
            "POST, /a2a",
            "GET,  /.well-known/agent.json"
    })
    void shouldNotFilter_exemptsThirdPartyProtocolEndpoints(String method, String uri) {
        assertThat(filter.shouldNotFilter(request(method, uri))).isTrue();
    }

    @ParameterizedTest(name = "{0} {1} 受保护")
    @DisplayName("受保护路径仍走本地鉴权")
    @CsvSource({
            "GET,  /api/v1/messages",
            "POST, /api/v1/messages/send",
            "POST, /api/v1/ai/conversations/1/messages",
            "POST, /api/v1/ai/skills/upload",
            "GET,  /api/v1/ai/models/enabled",
            "POST, /api/v1/ai/agents/5/a2a",
            "GET,  /api/v1/ai/mcp/servers/7/tools"
    })
    void shouldNotFilter_keepsProtectedPathsFiltered(String method, String uri) {
        assertThat(filter.shouldNotFilter(request(method, uri))).isFalse();
    }

    @Test
    @DisplayName("原有公开路径豁免行为不变")
    void shouldNotFilter_keepsExistingPublicPaths() {
        assertThat(filter.shouldNotFilter(request("GET", "/health"))).isTrue();
        assertThat(filter.shouldNotFilter(request("POST", "/api/v1/auth/login"))).isTrue();
        assertThat(filter.shouldNotFilter(request("GET", "/api/v1/auth/captcha"))).isTrue();
    }

    private static MockHttpServletRequest request(String method, String uri) {
        return new MockHttpServletRequest(method.trim(), uri.trim());
    }
}
