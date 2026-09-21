package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.exception.GlobalExceptionHandler;
import com.pei.dehaze.model.vo.AiArtifactVO;
import com.pei.dehaze.model.vo.KbChunkVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.AiArtifactService;
import com.pei.dehaze.service.AiKnowledgeBaseService;
import com.pei.dehaze.service.AiMcpService;
import com.pei.dehaze.service.AiModelService;
import com.pei.dehaze.service.AiProviderService;
import com.pei.dehaze.service.AiSkillService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import org.springframework.validation.beanvalidation.LocalValidatorFactoryBean;

import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * 分页参数边界校验（pageNum ≥ 1、pageSize ≤ 100，对齐 python BasePageQuery）。
 * <p>
 * 分页参数走 {@code BasePageQuery} 派生 DTO 后由 Bean Validation 承担边界校验，越界返回 400 + A0400；
 * 此前裸 {@code @RequestParam} 无 @Max，或 DTO 缺参数级 {@code @Valid}（注解成死注解），
 * 任意 pageSize 都会直达分页查询。
 */
@DisplayName("AI 域分页参数边界校验")
@ExtendWith(MockitoExtension.class)
class AiPagingParamValidationTest {

    private static final String PAGE_SIZE_MESSAGE = "每页大小不能超过100";

    @Mock
    private AiKnowledgeBaseService knowledgeBaseService;
    @Mock
    private AiArtifactService artifactService;
    @Mock
    private AiMcpService mcpService;
    @Mock
    private AiModelService modelService;
    @Mock
    private AiProviderService providerService;
    @Mock
    private AiSkillService skillService;

    private MockMvc kbMvc;
    private MockMvc artifactMvc;
    private MockMvc mcpMvc;
    private MockMvc modelMvc;
    private MockMvc providerMvc;
    private MockMvc skillMvc;

    @BeforeEach
    void setUp() {
        LocalValidatorFactoryBean validator = new LocalValidatorFactoryBean();
        validator.afterPropertiesSet();
        kbMvc = standalone(new AiKnowledgeBaseController(knowledgeBaseService), validator);
        artifactMvc = standalone(new AiArtifactController(artifactService), validator);
        mcpMvc = standalone(new AiMcpController(mcpService), validator);
        modelMvc = standalone(new AiModelController(modelService), validator);
        providerMvc = standalone(new AiProviderController(providerService), validator);
        skillMvc = standalone(new AiSkillController(skillService), validator);
    }

    @Test
    @DisplayName("分块列表 pageSize=101 返回 400 + A0400，且不触达服务层")
    void chunksPageSizeOverLimitRejected() throws Exception {
        kbMvc.perform(get("/api/v1/kb/documents/5/chunks").param("pageSize", "101"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"))
                .andExpect(jsonPath("$.msg").value(PAGE_SIZE_MESSAGE));

        verifyNoInteractions(knowledgeBaseService);
    }

    @Test
    @DisplayName("分块列表 pageNum=0 返回 400 + A0400")
    void chunksPageNumBelowMinRejected() throws Exception {
        kbMvc.perform(get("/api/v1/kb/documents/5/chunks").param("pageNum", "0"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"))
                .andExpect(jsonPath("$.msg").value("页码必须大于0"));

        verifyNoInteractions(knowledgeBaseService);
    }

    @Test
    @DisplayName("召回测试集与低质量片段列表 pageSize 超限同样 400")
    void otherKbListEndpointsRejectOversizePageSize() throws Exception {
        kbMvc.perform(get("/api/v1/kb/9/retrieve/test-sets").param("pageSize", "101"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));
        kbMvc.perform(get("/api/v1/kb/9/chunks/low-quality").param("pageSize", "101"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"));

        verifyNoInteractions(knowledgeBaseService);
    }

    @Test
    @DisplayName("知识库列表与文档列表（Query DTO 参数级 @Valid 生效）pageSize=101 返回 400 + A0400")
    void knowledgeBaseQueryDtosRejectOversizePageSize() throws Exception {
        kbMvc.perform(get("/api/v1/kb").param("pageSize", "101"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"))
                .andExpect(jsonPath("$.msg").value(PAGE_SIZE_MESSAGE));
        kbMvc.perform(get("/api/v1/kb/9/documents").param("pageSize", "101"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"))
                .andExpect(jsonPath("$.msg").value(PAGE_SIZE_MESSAGE));

        verifyNoInteractions(knowledgeBaseService);
    }

    @Test
    @DisplayName("MCP/模型/供应商/Skill 列表 pageSize=101 返回 400 + A0400，且不触达服务层")
    void aiAdminListEndpointsRejectOversizePageSize() throws Exception {
        mcpMvc.perform(get("/api/v1/ai/mcp/servers").param("pageSize", "101"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"))
                .andExpect(jsonPath("$.msg").value(PAGE_SIZE_MESSAGE));
        mcpMvc.perform(get("/api/v1/ai/mcp/calls").param("pageSize", "101"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"))
                .andExpect(jsonPath("$.msg").value(PAGE_SIZE_MESSAGE));
        modelMvc.perform(get("/api/v1/ai/models").param("pageSize", "101"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"))
                .andExpect(jsonPath("$.msg").value(PAGE_SIZE_MESSAGE));
        // 售价版本查询沿用 python 的 page/size 命名
        modelMvc.perform(get("/api/v1/ai/models/model-a/prices").param("size", "101"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"))
                .andExpect(jsonPath("$.msg").value(PAGE_SIZE_MESSAGE));
        providerMvc.perform(get("/api/v1/ai/providers").param("pageSize", "101"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"))
                .andExpect(jsonPath("$.msg").value(PAGE_SIZE_MESSAGE));
        skillMvc.perform(get("/api/v1/ai/skills").param("pageSize", "101"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"))
                .andExpect(jsonPath("$.msg").value(PAGE_SIZE_MESSAGE));

        verifyNoInteractions(mcpService, modelService, providerService, skillService);
    }

    @Test
    @DisplayName("会话产物列表 pageSize=101 返回 400 + A0400，且不触达服务层")
    void artifactPageSizeOverLimitRejected() throws Exception {
        artifactMvc.perform(get("/api/v1/ai/conversations/7/artifacts").param("pageSize", "101"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("A0400"))
                .andExpect(jsonPath("$.msg").value(PAGE_SIZE_MESSAGE));

        verifyNoInteractions(artifactService);
    }

    @Test
    @DisplayName("分块列表 pageSize=100 边界放行并透传 pageNum/pageSize")
    void chunksPageSizeAtLimitAllowed() throws Exception {
        when(knowledgeBaseService.listChunks(eq(5L), eq(1), eq(100)))
                .thenReturn(new Page<KbChunkVO>(1, 100, 0));

        kbMvc.perform(get("/api/v1/kb/documents/5/chunks").param("pageSize", "100"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code").value("00000"));

        verify(knowledgeBaseService).listChunks(5L, 1, 100);
    }

    @Test
    @DisplayName("会话产物列表 pageSize=100 边界放行并透传 pageNum/pageSize")
    void artifactPageSizeAtLimitAllowed() throws Exception {
        when(artifactService.listByConversation(eq(7L), eq(3L), eq(1), eq(100)))
                .thenReturn(new Page<AiArtifactVO>(1, 100, 0));

        try (MockedStatic<SecurityUtils> mocked = mockStatic(SecurityUtils.class)) {
            mocked.when(SecurityUtils::getUserId).thenReturn(3L);

            artifactMvc.perform(get("/api/v1/ai/conversations/7/artifacts").param("pageSize", "100"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.code").value("00000"));

            verify(artifactService).listByConversation(7L, 3L, 1, 100);
        }
    }

    private static MockMvc standalone(Object controller, LocalValidatorFactoryBean validator) {
        return MockMvcBuilders.standaloneSetup(controller)
                .setControllerAdvice(new GlobalExceptionHandler())
                .setValidator(validator)
                .build();
    }
}
