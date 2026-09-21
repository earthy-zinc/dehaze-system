package com.pei.dehaze.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysKnowledgeBaseMapper;
import com.pei.dehaze.mapper.SysKnowledgeChunkFeedbackMapper;
import com.pei.dehaze.mapper.SysKnowledgeChunkMapper;
import com.pei.dehaze.mapper.SysKnowledgeDocumentMapper;
import com.pei.dehaze.mapper.SysKnowledgeTestSetMapper;
import com.pei.dehaze.model.entity.SysKnowledgeBase;
import com.pei.dehaze.model.entity.SysKnowledgeChunk;
import com.pei.dehaze.model.entity.SysKnowledgeDocument;
import com.pei.dehaze.model.vo.KbChunkVO;
import com.pei.dehaze.security.util.SecurityUtils;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.data.redis.core.StringRedisTemplate;

import java.lang.reflect.Field;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * AI 知识库文档分块列表（GET /kb/documents/{id}/chunks）的可见性与分页测试。
 */
@DisplayName("AI 知识库文档分块列表测试")
@ExtendWith(MockitoExtension.class)
class AiKnowledgeBaseDocumentTest {

    private static final Long KB_ID = 9L;
    private static final Long DOC_ID = 5L;

    @Mock
    private SysKnowledgeDocumentMapper documentMapper;
    @Mock
    private SysKnowledgeTestSetMapper testSetMapper;
    @Mock
    private SysKnowledgeChunkMapper chunkMapper;
    @Mock
    private SysKnowledgeChunkFeedbackMapper chunkFeedbackMapper;
    @Mock
    private AiKbIndexClient kbIndexClient;
    @Mock
    private StringRedisTemplate redis;
    @Mock
    private SysKnowledgeBaseMapper kbMapper;

    private AiKnowledgeBaseServiceImpl service;

    @BeforeEach
    void setUp() throws Exception {
        service = new AiKnowledgeBaseServiceImpl(documentMapper, testSetMapper, chunkMapper,
                chunkFeedbackMapper, kbIndexClient, redis, new ObjectMapper());
        Field baseMapperField = ServiceImpl.class.getDeclaredField("baseMapper");
        baseMapperField.setAccessible(true);
        baseMapperField.set(service, kbMapper);
    }

    @Test
    @DisplayName("分块列表 - 越权查看他人私有库文档拒绝(A0301)，不查分块")
    void listChunks_otherUsersPrivateKbDenied() {
        when(kbMapper.selectById(KB_ID)).thenReturn(kb(2L));
        when(documentMapper.selectById(DOC_ID)).thenReturn(document());

        try (MockedStatic<SecurityUtils> mocked = mockStatic(SecurityUtils.class)) {
            mocked.when(SecurityUtils::getUserId).thenReturn(1L);

            BusinessException ex = assertThrows(BusinessException.class,
                    () -> service.listChunks(DOC_ID, 1, 10));

            assertEquals(ResultCode.ACCESS_UNAUTHORIZED.getCode(), ex.getResultCode().getCode());
            verifyNoInteractions(chunkMapper);
        }
    }

    @Test
    @DisplayName("分块列表 - 文档不存在返回 A0401")
    void listChunks_documentNotFound() {
        when(documentMapper.selectById(DOC_ID)).thenReturn(null);

        BusinessException ex = assertThrows(BusinessException.class,
                () -> service.listChunks(DOC_ID, 1, 10));

        assertEquals(ResultCode.RESOURCE_NOT_FOUND.getCode(), ex.getResultCode().getCode());
        verifyNoInteractions(chunkMapper);
    }

    @Test
    @DisplayName("分块列表 - 分页返回并按 chunk_index 升序，metadata 解析为对象")
    void listChunks_returnsPagedChunks() {
        when(kbMapper.selectById(KB_ID)).thenReturn(kb(1L));
        when(documentMapper.selectById(DOC_ID)).thenReturn(document());
        Page<SysKnowledgeChunk> chunkPage = new Page<>(1, 10, 1);
        chunkPage.setRecords(List.of(chunk()));
        when(chunkMapper.selectPage(any(Page.class), any(LambdaQueryWrapper.class))).thenReturn(chunkPage);

        try (MockedStatic<SecurityUtils> mocked = mockStatic(SecurityUtils.class)) {
            mocked.when(SecurityUtils::getUserId).thenReturn(1L);

            Page<KbChunkVO> result = service.listChunks(DOC_ID, 1, 10);

            assertThat(result.getTotal()).isEqualTo(1);
            KbChunkVO vo = result.getRecords().get(0);
            assertThat(vo.getId()).isEqualTo(31L);
            assertThat(vo.getDocumentId()).isEqualTo(DOC_ID);
            assertThat(vo.getChunkIndex()).isEqualTo(2);
            assertThat(vo.getContent()).isEqualTo("分块内容");
            assertThat(vo.getTokenCount()).isEqualTo(128);
            assertThat(vo.getMetadata()).isEqualTo(Map.of("page", 2, "type", "text"));
        }
    }

    private static SysKnowledgeBase kb(Long ownerId) {
        SysKnowledgeBase kb = new SysKnowledgeBase();
        kb.setId(KB_ID);
        kb.setVisibility("private");
        kb.setCreateBy(ownerId);
        return kb;
    }

    private static SysKnowledgeDocument document() {
        SysKnowledgeDocument doc = new SysKnowledgeDocument();
        doc.setId(DOC_ID);
        doc.setKnowledgeBaseId(KB_ID);
        doc.setProcessingStatus("completed");
        doc.setVersion(1);
        return doc;
    }

    private static SysKnowledgeChunk chunk() {
        SysKnowledgeChunk chunk = new SysKnowledgeChunk();
        chunk.setId(31L);
        chunk.setDocumentId(DOC_ID);
        chunk.setKnowledgeBaseId(KB_ID);
        chunk.setChunkIndex(2);
        chunk.setContent("分块内容");
        chunk.setTokenCount(128);
        chunk.setMetadata("{\"page\":2,\"type\":\"text\"}");
        return chunk;
    }
}
