package com.pei.dehaze.service.impl;

import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.common.util.AiCredentialCipher;
import com.pei.dehaze.mapper.SysAiMcpCallMapper;
import com.pei.dehaze.mapper.SysAiMcpNamespaceMapper;
import com.pei.dehaze.mapper.SysAiMcpServerMapper;
import com.pei.dehaze.mapper.SysAiMcpToolMapper;
import com.pei.dehaze.model.entity.SysAiMcpServer;
import com.pei.dehaze.model.form.McpServerForm;
import com.pei.dehaze.model.form.McpServerUpdateForm;
import com.pei.dehaze.model.vo.McpServerVO;
import com.pei.dehaze.service.AiCacheInvalidator;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.NullSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.test.util.ReflectionTestUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * MCP {@code protocol_type} 白名单（service 层单一信息源，对齐 python {@code Literal["streamable-http","sse"]}）。
 *
 * <p>python 由 pydantic 在请求校验阶段拒绝非法字面量（A0400）；java 若不在 service 层拦，
 * 拼错的协议名会直接落库，后续探测/装载阶段才静默失败（脏数据 + 难定位）。
 */
@DisplayName("AiMcpServiceImpl protocolType 白名单")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AiMcpProtocolWhitelistTest {

    @Mock
    private SysAiMcpServerMapper serverMapper;

    @Mock
    private SysAiMcpNamespaceMapper namespaceMapper;

    @Mock
    private SysAiMcpToolMapper toolMapper;

    @Mock
    private SysAiMcpCallMapper callMapper;

    @Mock
    private AiCredentialCipher cipher;

    @Mock
    private AiCacheInvalidator cacheInvalidator;

    private AiMcpServiceImpl service;

    @BeforeEach
    void setUp() {
        service = new AiMcpServiceImpl(namespaceMapper, toolMapper, callMapper, cipher, cacheInvalidator);
        ReflectionTestUtils.setField(service, "baseMapper", serverMapper);
    }

    @ParameterizedTest(name = "protocolType=[{0}] → A0400 且不触达 mapper")
    @NullSource
    @ValueSource(strings = {"stdio", "websocket", "STREAMABLE-HTTP", "streamable_http", "sse ", "", "sse;sse"})
    @DisplayName("createServer：白名单外协议一律 A0400，不得落库")
    void createServerRejectsProtocolOutsideWhitelist(String protocolType) {
        McpServerForm form = new McpServerForm();
        form.setName("mcp_probe");
        form.setEndpoint("https://example.com/mcp");
        form.setProtocolType(protocolType);

        BusinessException ex = assertThrows(BusinessException.class, () -> service.createServer(form));

        assertThat(ex.getResultCode()).isEqualTo(ResultCode.PARAM_ERROR);
        verifyNoInteractions(serverMapper, namespaceMapper, toolMapper, callMapper, cacheInvalidator);
    }

    @Test
    @DisplayName("updateServer：白名单外协议 A0400，且先于 Server 存在性查询")
    void updateServerRejectsProtocolOutsideWhitelist() {
        McpServerUpdateForm form = new McpServerUpdateForm();
        form.setProtocolType("stdio");

        BusinessException ex = assertThrows(BusinessException.class, () -> service.updateServer(7L, form));

        assertThat(ex.getResultCode()).isEqualTo(ResultCode.PARAM_ERROR);
        verifyNoInteractions(serverMapper);
    }

    @Test
    @DisplayName("白名单内协议放行：sse 落库并触发推理图缓存失效广播")
    void updateServerAcceptsWhitelistedProtocol() {
        SysAiMcpServer server = new SysAiMcpServer();
        server.setId(7L);
        server.setName("mcp_a");
        server.setProtocolType("streamable-http");
        when(serverMapper.selectById(7L)).thenReturn(server);

        McpServerUpdateForm form = new McpServerUpdateForm();
        form.setProtocolType("sse");

        McpServerVO vo = service.updateServer(7L, form);

        assertThat(vo.getProtocolType()).isEqualTo("sse");
        verify(serverMapper).updateById(any());
        verify(cacheInvalidator).evictReasoningGraphs();
    }
}
