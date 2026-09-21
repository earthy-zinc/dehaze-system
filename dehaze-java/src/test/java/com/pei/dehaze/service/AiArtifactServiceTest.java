package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.conditions.Wrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysAiArtifactMapper;
import com.pei.dehaze.mapper.SysEvalLogMapper;
import com.pei.dehaze.mapper.SysFileMapper;
import com.pei.dehaze.mapper.SysPredLogMapper;
import com.pei.dehaze.model.entity.SysAiArtifact;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.model.entity.SysPredLog;
import com.pei.dehaze.service.impl.file.StorageServiceFactory;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.function.Executable;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * AI 中间产物服务单测：产物归属随会话校验、按引用反查只返回本人会话产物、图片 URL 运行时按 ref 链路拼接。
 *
 * <p>产物含他人会话的图片引用，归属校验缺失即为横向越权读；URL 遵循"永不落库"，
 * 由 {@code sys_file / sys_pred_log / sys_eval_log} 链路在响应期拼接。
 */
@DisplayName("AiArtifactService 产物归属与 URL 拼接")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AiArtifactServiceTest {

    private static final Long USER_ID = 3L;

    @Mock
    private SysAiArtifactMapper artifactMapper;
    @Mock
    private SysFileMapper fileMapper;
    @Mock
    private SysPredLogMapper predLogMapper;
    @Mock
    private SysEvalLogMapper evalLogMapper;
    @Mock
    private StorageServiceFactory storageServiceFactory;
    @Mock
    private AiConversationService conversationService;

    @InjectMocks
    private AiArtifactService service;

    private SysAiArtifact artifact(Long id, Long convId, String refType, Long refId) {
        SysAiArtifact artifact = new SysAiArtifact();
        artifact.setId(id);
        artifact.setConversationId(convId);
        artifact.setMessageId(9L);
        artifact.setType("image");
        artifact.setRefType(refType);
        artifact.setRefId(refId);
        artifact.setSummary(Map.of("text", "去雾结果"));
        artifact.setIsInvalid(0);
        return artifact;
    }

    private void assertBizError(ResultCode expected, Executable action) {
        assertThat(assertThrows(BusinessException.class, action).getResultCode()).isEqualTo(expected);
    }

    @Test
    @DisplayName("会话产物列表：归属校验失败（他人会话）直接报错，不返回任何数据")
    void listByConversationChecksOwnership() {
        when(conversationService.getOwned(1L, USER_ID))
                .thenThrow(new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "会话不存在"));

        assertBizError(ResultCode.RESOURCE_NOT_FOUND, () -> service.listByConversation(1L, USER_ID, 1, 10));
        verify(artifactMapper, never()).selectPage(any(), any(Wrapper.class));
    }

    @Test
    @DisplayName("会话产物列表：分页结构透传并映射核心字段")
    void listByConversationMapsFields() {
        Page<SysAiArtifact> page = new Page<>(1, 10, 1);
        page.setRecords(List.of(artifact(5L, 1L, "sys_file", 77L)));
        when(artifactMapper.selectPage(any(), any(Wrapper.class))).thenReturn(page);

        var result = service.listByConversation(1L, USER_ID, 1, 10);

        assertThat(result.getTotal()).isEqualTo(1);
        assertThat(result.getRecords()).hasSize(1);
        assertThat(result.getRecords().get(0).getRefType()).isEqualTo("sys_file");
        assertThat(result.getRecords().get(0).getRefId()).isEqualTo(77L);
        verify(conversationService).getOwned(1L, USER_ID);
    }

    @Test
    @DisplayName("消息产物列表：消息不存在或非本人报 A0401")
    void listByMessageChecksMessageOwnership() {
        when(conversationService.getOwnedMessage(9L, USER_ID)).thenReturn(null);

        assertBizError(ResultCode.RESOURCE_NOT_FOUND, () -> service.listByMessage(9L, USER_ID));
        verify(artifactMapper, never()).selectList(any(Wrapper.class));
    }

    @Test
    @DisplayName("按引用反查：只返回属于当前用户会话的产物（他人会话产物被过滤）")
    void listByRefFiltersForeignConversations() {
        when(artifactMapper.selectList(any(Wrapper.class))).thenReturn(List.of(
                artifact(5L, 1L, "sys_pred_log", 88L),
                artifact(6L, 2L, "sys_pred_log", 88L)));
        when(conversationService.listOwnedIds(Set.of(1L, 2L), USER_ID)).thenReturn(Set.of(1L));

        assertThat(service.listByRef("sys_pred_log", 88L, USER_ID))
                .extracting(vo -> vo.getId()).containsExactly(5L);
    }

    @Test
    @DisplayName("产物详情：不存在或已失效报 A0401")
    void detailRejectsMissingOrInvalidArtifact() {
        when(artifactMapper.selectById(5L)).thenReturn(null);
        assertBizError(ResultCode.RESOURCE_NOT_FOUND, () -> service.getDetail(5L, USER_ID));

        SysAiArtifact invalid = artifact(5L, 1L, "sys_file", 77L);
        invalid.setIsInvalid(1);
        when(artifactMapper.selectById(5L)).thenReturn(invalid);
        assertBizError(ResultCode.RESOURCE_NOT_FOUND, () -> service.getDetail(5L, USER_ID));
    }

    @Test
    @DisplayName("产物详情：sys_file 引用直接取文件，URL 经存储后端运行时拼接")
    void detailResolvesImageUrlFromFile() {
        when(artifactMapper.selectById(5L)).thenReturn(artifact(5L, 1L, "sys_file", 77L));
        SysFile file = new SysFile();
        file.setId(77L);
        file.setStorage("local");
        file.setObjectName("images/a.jpg");
        when(fileMapper.selectById(77L)).thenReturn(file);
        FileService storage = mock(FileService.class);
        when(storageServiceFactory.get("local")).thenReturn(storage);
        when(storage.getUrl("images/a.jpg")).thenReturn("https://cdn/images/a.jpg");

        Map<String, Object> detail = service.getDetail(5L, USER_ID);

        assertThat(detail.get("imageUrl")).isEqualTo("https://cdn/images/a.jpg");
        verify(conversationService).getOwned(1L, USER_ID);
    }

    @Test
    @DisplayName("产物详情：sys_pred_log 引用经预测记录取文件；链路缺失时 URL 为空而非报错")
    void detailResolvesImageUrlFromPredLogChain() {
        when(artifactMapper.selectById(5L)).thenReturn(artifact(5L, 1L, "sys_pred_log", 88L));
        when(predLogMapper.selectById(88L)).thenReturn(null);

        assertThat(service.getDetail(5L, USER_ID).get("imageUrl")).isNull();

        SysPredLog pred = new SysPredLog();
        pred.setPredFileId(77L);
        when(predLogMapper.selectById(88L)).thenReturn(pred);
        when(fileMapper.selectById(77L)).thenReturn(null);

        assertThat(service.getDetail(5L, USER_ID).get("imageUrl")).isNull();
        verify(fileMapper).selectById(77L);
        verify(storageServiceFactory, never()).get(any());
    }
}
