package com.pei.dehaze.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysAiMemoryMapper;
import com.pei.dehaze.mapper.SysAiMessageFeedbackMapper;
import com.pei.dehaze.model.entity.SysAiMemory;
import com.pei.dehaze.model.entity.SysAiMessage;
import com.pei.dehaze.model.entity.SysAiMessageFeedback;
import com.pei.dehaze.model.form.AiFeedbackCreateForm;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.function.Executable;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.Spy;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * AI 消息反馈服务单测：仅助手消息可反馈、30 天时效、标签白名单、软删行复活、点踩沉淀偏好记忆。
 *
 * <p>对齐 dehaze-python {@code ai_feedback_service}：(message_id,user_id) 唯一键 upsert（含软删行复活），
 * 点踩标签映射为常驻偏好记忆（source=feedback / is_preference=1），供后续会话注入生效。
 */
@DisplayName("AiFeedbackService 消息反馈")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AiFeedbackServiceTest {

    private static final Long USER_ID = 3L;
    private static final Long MESSAGE_ID = 9L;

    @Mock
    private SysAiMessageFeedbackMapper feedbackMapper;
    @Mock
    private SysAiMemoryMapper memoryMapper;
    @Mock
    private AiConversationService conversationService;

    @Spy
    private ObjectMapper objectMapper = new ObjectMapper();

    @InjectMocks
    private AiFeedbackService service;

    private SysAiMessage message(String role, LocalDateTime createTime) {
        SysAiMessage message = new SysAiMessage();
        message.setId(MESSAGE_ID);
        message.setConversationId(1L);
        message.setRole(role);
        message.setModel("qwen3-0.6b");
        message.setCreateTime(createTime);
        return message;
    }

    private AiFeedbackCreateForm form(Integer rating, List<String> tags, String comment) {
        AiFeedbackCreateForm form = new AiFeedbackCreateForm();
        form.setRating(rating);
        form.setTags(tags);
        form.setComment(comment);
        return form;
    }

    private void assertBizError(ResultCode expected, Executable action) {
        assertThat(assertThrows(BusinessException.class, action).getResultCode()).isEqualTo(expected);
    }

    @Test
    @DisplayName("仅助手消息可反馈：用户消息报 A0502")
    void feedbackOnUserMessageRejected() {
        when(conversationService.getOwnedMessage(MESSAGE_ID, USER_ID))
                .thenReturn(message("user", LocalDateTime.now()));

        assertBizError(ResultCode.DATA_STATE_NOT_ALLOW,
                () -> service.submit(MESSAGE_ID, USER_ID, form(1, List.of("accurate"), null)));
        verify(feedbackMapper, never()).insert(any(SysAiMessageFeedback.class));
    }

    @Test
    @DisplayName("他人消息不可见：归属校验失败报 A0401")
    void feedbackOnForeignMessageRejected() {
        when(conversationService.getOwnedMessage(MESSAGE_ID, USER_ID)).thenReturn(null);

        assertBizError(ResultCode.RESOURCE_NOT_FOUND,
                () -> service.submit(MESSAGE_ID, USER_ID, form(1, null, null)));
    }

    @Test
    @DisplayName("超过 30 天反馈时效报 A0502")
    void feedbackExpiredRejected() {
        when(conversationService.getOwnedMessage(MESSAGE_ID, USER_ID))
                .thenReturn(message("assistant", LocalDateTime.now().minusDays(31)));

        assertBizError(ResultCode.DATA_STATE_NOT_ALLOW,
                () -> service.submit(MESSAGE_ID, USER_ID, form(1, List.of("accurate"), null)));
    }

    @Test
    @DisplayName("点赞标签必须属于点赞白名单")
    void likeRejectsDislikeTag() {
        when(conversationService.getOwnedMessage(MESSAGE_ID, USER_ID))
                .thenReturn(message("assistant", LocalDateTime.now()));

        assertBizError(ResultCode.PARAM_ERROR,
                () -> service.submit(MESSAGE_ID, USER_ID, form(1, List.of("incorrect"), null)));
    }

    @Test
    @DisplayName("点踩必须选择问题标签，且标签须在白名单内")
    void dislikeRequiresKnownTag() {
        when(conversationService.getOwnedMessage(MESSAGE_ID, USER_ID))
                .thenReturn(message("assistant", LocalDateTime.now()));

        // 点踩取 -1（python Literal[1,-1]）；若沿用 0 会被评分白名单先拦，本用例将测不到标签规则（假绿）
        assertBizError(ResultCode.PARAM_ERROR,
                () -> service.submit(MESSAGE_ID, USER_ID, form(-1, List.of(), null)));
        assertBizError(ResultCode.PARAM_ERROR,
                () -> service.submit(MESSAGE_ID, USER_ID, form(-1, List.of("whatever"), null)));
    }

    @Test
    @DisplayName("点赞可无标签，落库为内部来源")
    void likeWithoutTagInsertsInternalFeedback() {
        when(conversationService.getOwnedMessage(MESSAGE_ID, USER_ID))
                .thenReturn(message("assistant", LocalDateTime.now()));
        when(feedbackMapper.selectByUserAndMessageIgnoringDeleted(MESSAGE_ID, USER_ID)).thenReturn(null);

        service.submit(MESSAGE_ID, USER_ID, form(1, null, "不错"));

        ArgumentCaptor<SysAiMessageFeedback> captor = ArgumentCaptor.forClass(SysAiMessageFeedback.class);
        verify(feedbackMapper).insert(captor.capture());
        assertThat(captor.getValue().getSource()).isEqualTo("internal");
        assertThat(captor.getValue().getProcessed()).isZero();
        assertThat(captor.getValue().getConversationId()).isEqualTo(1L);
    }

    @Test
    @DisplayName("已存在的反馈（含软删行）走复活更新，不新增行（唯一键 (message_id,user_id)）")
    void existingFeedbackIsRevived() {
        when(conversationService.getOwnedMessage(MESSAGE_ID, USER_ID))
                .thenReturn(message("assistant", LocalDateTime.now()));
        SysAiMessageFeedback existing = new SysAiMessageFeedback();
        existing.setId(21L);
        existing.setDeleted(1L);
        when(feedbackMapper.selectByUserAndMessageIgnoringDeleted(MESSAGE_ID, USER_ID)).thenReturn(existing);
        // 复活后服务会重新读取该行返回给调用方
        when(feedbackMapper.selectById(21L)).thenReturn(existing);

        service.submit(MESSAGE_ID, USER_ID, form(1, List.of("accurate"), null));

        // 标签以 JSON 数组字符串落库（与 python 存储口径一致）
        verify(feedbackMapper).revive(eq(21L), eq(1), eq("[\"accurate\"]"), isNull(), eq(1L),
                eq("qwen3-0.6b"), eq("internal"));
        verify(feedbackMapper, never()).insert(any(SysAiMessageFeedback.class));
    }

    @Test
    @DisplayName("点踩沉淀偏好记忆：source=feedback / importance=100 / is_preference=1，并带上用户补充")
    void dislikePersistsPreferenceMemory() {
        when(conversationService.getOwnedMessage(MESSAGE_ID, USER_ID))
                .thenReturn(message("assistant", LocalDateTime.now()));
        when(feedbackMapper.selectByUserAndMessageIgnoringDeleted(MESSAGE_ID, USER_ID)).thenReturn(null);

        // 点踩在 python 侧是 Literal[1,-1] 中的 -1（0 不是合法取值，旧用例按 0 构造已与契约不符）
        service.submit(MESSAGE_ID, USER_ID, form(-1, List.of("too_long", "incorrect"), "太啰嗦了"));

        ArgumentCaptor<SysAiMemory> captor = ArgumentCaptor.forClass(SysAiMemory.class);
        verify(memoryMapper).insert(captor.capture());
        SysAiMemory memory = captor.getValue();
        assertThat(memory.getSource()).isEqualTo("feedback");
        assertThat(memory.getImportance()).isEqualTo(100);
        assertThat(memory.getMemoryType()).isEqualTo("semantic");
        assertThat(memory.getContent()).contains("用户偏好简洁回复").contains("太啰嗦了");
        Map<?, ?> metadata = (Map<?, ?>) memory.getMetadata();
        assertThat(metadata.get("is_preference")).isEqualTo(1);
        assertThat(metadata.get("category")).isEqualTo("preference");
    }

    @Test
    @DisplayName("点踩但标签无偏好映射时不沉淀记忆")
    void dislikeWithoutMappedTagSkipsMemory() {
        when(conversationService.getOwnedMessage(MESSAGE_ID, USER_ID))
                .thenReturn(message("assistant", LocalDateTime.now()));
        when(feedbackMapper.selectByUserAndMessageIgnoringDeleted(MESSAGE_ID, USER_ID)).thenReturn(null);

        service.submit(MESSAGE_ID, USER_ID, form(-1, List.of("harmful"), null));

        verify(memoryMapper, never()).insert(any(SysAiMemory.class));
    }

    @Test
    @DisplayName("点赞不沉淀偏好记忆")
    void likeDoesNotPersistMemory() {
        when(conversationService.getOwnedMessage(MESSAGE_ID, USER_ID))
                .thenReturn(message("assistant", LocalDateTime.now()));
        when(feedbackMapper.selectByUserAndMessageIgnoringDeleted(MESSAGE_ID, USER_ID)).thenReturn(null);

        service.submit(MESSAGE_ID, USER_ID, form(1, List.of("detailed"), null));

        verify(memoryMapper, never()).insert(any(SysAiMemory.class));
    }

    @Test
    @DisplayName("查询反馈：已撤销（软删）返回空")
    void getReturnsNullForRevokedFeedback() {
        SysAiMessageFeedback revoked = new SysAiMessageFeedback();
        revoked.setId(21L);
        revoked.setDeleted(1L);
        when(feedbackMapper.selectByUserAndMessageIgnoringDeleted(MESSAGE_ID, USER_ID)).thenReturn(revoked);

        assertThat(service.get(MESSAGE_ID, USER_ID)).isNull();
    }

    @Test
    @DisplayName("撤销反馈：无反馈报 A0543，存在则软删")
    void revokeSoftDeletesExistingFeedback() {
        when(feedbackMapper.selectByUserAndMessageIgnoringDeleted(MESSAGE_ID, USER_ID)).thenReturn(null);
        assertBizError(ResultCode.FEEDBACK_NOT_FOUND, () -> service.revoke(MESSAGE_ID, USER_ID));

        SysAiMessageFeedback existing = new SysAiMessageFeedback();
        existing.setId(21L);
        existing.setDeleted(0L);
        existing.setTags(List.of("accurate"));
        when(feedbackMapper.selectByUserAndMessageIgnoringDeleted(MESSAGE_ID, USER_ID)).thenReturn(existing);

        service.revoke(MESSAGE_ID, USER_ID);

        verify(feedbackMapper).softDeleteByUserAndMessage(MESSAGE_ID, USER_ID);
        verify(memoryMapper, never()).insert(any(SysAiMemory.class));
        verify(feedbackMapper, never()).insert(any(SysAiMessageFeedback.class));
    }

    @Test
    @DisplayName("查询反馈：返回标签列表")
    void getReturnsTags() {
        SysAiMessageFeedback existing = new SysAiMessageFeedback();
        existing.setId(21L);
        existing.setDeleted(0L);
        existing.setMessageId(MESSAGE_ID);
        existing.setUserId(USER_ID);
        existing.setRating(1);
        existing.setTags(List.of("accurate", "detailed"));
        when(feedbackMapper.selectByUserAndMessageIgnoringDeleted(MESSAGE_ID, USER_ID)).thenReturn(existing);

        assertThat(service.get(MESSAGE_ID, USER_ID).getTags()).containsExactly("accurate", "detailed");
    }

    @Test
    @DisplayName("反馈持久化未触发记忆写入（点赞路径）")
    void likePathDoesNotTouchMemoryMapper() {
        when(conversationService.getOwnedMessage(MESSAGE_ID, USER_ID))
                .thenReturn(message("assistant", LocalDateTime.now()));
        when(feedbackMapper.selectByUserAndMessageIgnoringDeleted(MESSAGE_ID, USER_ID)).thenReturn(null);

        service.submit(MESSAGE_ID, USER_ID, form(1, List.of("concise"), null));

        verify(memoryMapper, never()).insert(any(SysAiMemory.class));
        verify(memoryMapper, never()).touch(anyLong());
    }
}
