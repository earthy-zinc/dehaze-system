package com.pei.dehaze.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.common.util.AiCredentialCipher;
import com.pei.dehaze.mapper.SysAiAgentEndpointMapper;
import com.pei.dehaze.model.entity.SysAiAgentEndpoint;
import com.pei.dehaze.model.form.AiEndpointCreateForm;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;

import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * 外部 A2A 端点服务单测：SSRF 白名单（https + 非内网，含 IPv6 与 DNS 解析口径）与软删行复活。
 *
 * <p>SSRF 判定是「管理员注册外部端点」这条链路上唯一的安全边界，必须与 python {@code utils/ssrf}
 * 同口径：IPv6 字面量、ULA、IPv4 映射地址都不能放行。
 */
@DisplayName("AiAgentEndpointService 端点注册与 SSRF 防护")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AiAgentEndpointServiceTest {

    @Mock
    private SysAiAgentEndpointMapper endpointMapper;

    @Mock
    private AiCredentialCipher credentialCipher;

    @Mock
    private AuditLogService auditLogService;

    private AiAgentEndpointService service;

    @BeforeEach
    void setUp() {
        service = new AiAgentEndpointService(endpointMapper, credentialCipher, auditLogService,
                new ObjectMapper());
    }

    /**
     * 放行侧必须精确到段：同 /16 内的相邻段（198.51.99.x、198.20.x、192.0.3.x、203.0.114.x）与
     * 6to4 relay anycast（192.88.99.1，python is_global=True）都不能被误封——否则将来有人
     * 把 198.51.100.0/24 图省事写成 198.51.0.0/16 时不会有测试报警。
     */
    @ParameterizedTest(name = "公网字面地址放行：{0}")
    @ValueSource(strings = {"https://8.8.8.8/card", "https://1.1.1.1:8443/.well-known/agent.json",
            "https://[2606:4700:4700::1111]/card", "https://198.51.99.1/card",
            "https://198.20.0.1/card", "https://192.0.3.1/card", "https://203.0.114.1/card",
            "https://192.88.99.1/card"})
    void publicAddressAllowed(String url) {
        assertThat(AiAgentEndpointService.isSafeUrl(url)).isTrue();
    }

    @ParameterizedTest(name = "非 https 拒绝：{0}")
    @ValueSource(strings = {"http://8.8.8.8/card", "ftp://8.8.8.8/card", "not-a-url", "", " https://x"})
    void nonHttpsRejected(String url) {
        assertThat(AiAgentEndpointService.isSafeUrl(url)).isFalse();
    }

    @ParameterizedTest(name = "内网/环回/保留地址拒绝：{0}")
    @ValueSource(strings = {"https://localhost/card", "https://LOCALHOST/card", "https://svc.local/card",
            "https://meta.internal/card", "https://127.0.0.1/card", "https://10.1.2.3/card",
            "https://172.16.0.1/card", "https://172.31.255.254/card", "https://192.168.1.1/card",
            "https://169.254.169.254/latest/meta-data", "https://100.64.0.1/card",
            "https://0.0.0.0/card", "https://224.0.0.1/card", "https://240.0.0.1/card",
            "https://192.0.2.1/card", "https://198.51.100.1/card", "https://203.0.113.9/card",
            "https://198.18.0.1/card", "https://198.19.255.1/card",
            "https://[::1]/card", "https://[fc00::1]/card", "https://[fd12:3456::1]/card",
            "https://[fe80::1]/card", "https://[::ffff:127.0.0.1]/card"})
    void internalAddressRejected(String url) {
        assertThat(AiAgentEndpointService.isSafeUrl(url)).isFalse();
    }

    @Test
    @DisplayName("注册：base_url 非 https 或内网报 A0400，不落库")
    void createRejectsUnsafeBaseUrl() {
        AiEndpointCreateForm form = new AiEndpointCreateForm();
        form.setName("内网端点");
        form.setBaseUrl("http://10.0.0.9/a2a");

        BusinessException ex = assertThrows(BusinessException.class, () -> service.create(form));
        assertThat(ex.getResultCode()).isEqualTo(ResultCode.PARAM_ERROR);
        verify(endpointMapper, never()).insert(any(SysAiAgentEndpoint.class));
    }

    @Test
    @DisplayName("注册：agent_card_url 指向内网同样拒绝")
    void createRejectsInternalAgentCardUrl() {
        AiEndpointCreateForm form = new AiEndpointCreateForm();
        form.setName("外部端点");
        form.setBaseUrl("https://8.8.8.8/a2a");
        form.setAgentCardUrl("https://169.254.169.254/latest/meta-data");

        assertThat(assertThrows(BusinessException.class, () -> service.create(form)).getResultCode())
                .isEqualTo(ResultCode.PARAM_ERROR);
        verify(endpointMapper, never()).insert(any(SysAiAgentEndpoint.class));
    }

    @Test
    @DisplayName("注册：地址已存在且未删报 A0501")
    void createRejectsActiveDuplicate() {
        AiEndpointCreateForm form = new AiEndpointCreateForm();
        form.setName("外部端点");
        form.setBaseUrl("https://8.8.8.8/a2a");
        SysAiAgentEndpoint existing = new SysAiAgentEndpoint();
        existing.setId(7L);
        existing.setDeleted(0L);
        when(endpointMapper.selectByBaseUrlIgnoringDeleted("https://8.8.8.8/a2a")).thenReturn(existing);

        assertThat(assertThrows(BusinessException.class, () -> service.create(form)).getResultCode())
                .isEqualTo(ResultCode.DATA_EXISTS);
        verify(endpointMapper, never()).insert(any(SysAiAgentEndpoint.class));
    }

    @Test
    @DisplayName("注册：命中软删行则复活原行并覆盖为新表单值（唯一键含 deleted，不复用新行）")
    void createRevivesSoftDeletedRow() {
        AiEndpointCreateForm form = new AiEndpointCreateForm();
        form.setName("重新注册");
        form.setBaseUrl("https://8.8.8.8/a2a");
        form.setAuthType("http");
        SysAiAgentEndpoint existing = new SysAiAgentEndpoint();
        existing.setId(7L);
        existing.setDeleted(7L);
        when(endpointMapper.selectByBaseUrlIgnoringDeleted("https://8.8.8.8/a2a")).thenReturn(existing);

        assertThat(service.create(form).getId()).isEqualTo(7L);
        assertThat(existing.getDeleted()).isZero();
        assertThat(existing.getName()).isEqualTo("重新注册");
        verify(endpointMapper).updateById(existing);
        verify(endpointMapper, never()).insert(any(SysAiAgentEndpoint.class));
    }

    @Test
    @DisplayName("删除：软删端点并留审计（记录 name 与 base_url）")
    void deleteSoftDeletesAndAudits() {
        SysAiAgentEndpoint endpoint = new SysAiAgentEndpoint();
        endpoint.setId(7L);
        endpoint.setName("外部端点");
        endpoint.setBaseUrl("https://8.8.8.8/a2a");
        when(endpointMapper.selectById(7L)).thenReturn(endpoint);

        service.delete(7L, 1L);

        verify(endpointMapper).softDeleteByIds(java.util.List.of(7L));
        ArgumentCaptor<Object> beforeCaptor = ArgumentCaptor.forClass(Object.class);
        verify(auditLogService).recordAudit(eq(1L), eq("ai_agent_endpoint"), eq(7L), eq("delete"),
                eq("ai_agent"), beforeCaptor.capture(), isNull(), isNull(), isNull());
        Map<?, ?> beforeValue = (Map<?, ?>) beforeCaptor.getValue();
        assertThat(String.valueOf(beforeValue.get("base_url"))).isEqualTo("https://8.8.8.8/a2a");
        assertThat(String.valueOf(beforeValue.get("name"))).isEqualTo("外部端点");
    }
}
