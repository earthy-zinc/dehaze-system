package com.pei.dehaze.service.impl;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.baomidou.mybatisplus.core.toolkit.GlobalConfigUtils;
import com.pei.dehaze.common.util.AiCredentialCipher;
import com.pei.dehaze.mapper.SysAiMcpCallMapper;
import com.pei.dehaze.mapper.SysAiMcpNamespaceMapper;
import com.pei.dehaze.mapper.SysAiMcpServerMapper;
import com.pei.dehaze.mapper.SysAiMcpToolMapper;
import com.pei.dehaze.model.entity.SysAiMcpServer;
import com.pei.dehaze.model.entity.SysAiMcpTool;
import com.pei.dehaze.model.form.McpNamespaceForm;
import com.pei.dehaze.service.AiCacheInvalidator;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * MCP Server 配置变更的跨端失效联动。
 *
 * <p>推理图由 python 构建并缓存在进程内，java 原生改 MCP（共享库）不会触发其重建，
 * 已构图仍持旧的外部工具集，必须广播 cache:invalidation 让 python 失效自己的图缓存。
 */
@DisplayName("AiMcpServiceImpl 变更联动推理图缓存失效广播")
@ExtendWith(MockitoExtension.class)
class AiMcpServiceImplTest {

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

    /** 无 Spring 上下文时 LambdaQueryWrapper 的 select(...) 需要先注册实体表信息 */
    @BeforeAll
    static void initTableInfo() {
        MybatisConfiguration configuration = new MybatisConfiguration();
        GlobalConfigUtils.setGlobalConfig(configuration, GlobalConfigUtils.defaults());
        TableInfoHelper.initTableInfo(new MapperBuilderAssistant(configuration, ""), SysAiMcpTool.class);
    }

    @BeforeEach
    void setUp() {
        service = new AiMcpServiceImpl(namespaceMapper, toolMapper, callMapper, cipher, cacheInvalidator);
        ReflectionTestUtils.setField(service, "baseMapper", serverMapper);
    }

    @Test
    @DisplayName("命名空间覆盖式更新后广播推理图缓存失效")
    void updateNamespacesBroadcastsReasoningGraphInvalidation() {
        SysAiMcpServer server = new SysAiMcpServer();
        server.setId(7L);
        server.setName("mcp_a");
        when(serverMapper.selectById(7L)).thenReturn(server);
        SysAiMcpTool tool = new SysAiMcpTool();
        tool.setName("tool_a");
        when(toolMapper.selectList(any())).thenReturn(List.of(tool));
        when(namespaceMapper.selectList(any())).thenReturn(List.of());

        McpNamespaceForm form = new McpNamespaceForm();
        form.setName("ns_a");
        form.setToolNames(List.of("tool_a"));

        service.updateNamespaces(7L, List.of(form));

        verify(cacheInvalidator).evictReasoningGraphs();
    }
}
