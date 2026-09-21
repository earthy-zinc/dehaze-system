package com.pei.dehaze.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysAiMemoryMapper;
import com.pei.dehaze.mapper.SysAiMessageFeedbackMapper;
import com.pei.dehaze.model.entity.SysAiMessage;
import com.pei.dehaze.model.form.AiFeedbackCreateForm;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.Spy;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;

import java.time.LocalDateTime;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * 消息反馈评分白名单（service 层单一信息源，对齐 python {@code Literal[1,-1]}）。
 *
 * <p>为何不能用 {@code @Min(-1)@Max(1)}：合法集 {1,-1} **不连续**（0 是洞），区间注解必然放行 0；
 * 而原实现按 {@code equals(1)} 二分会把 0/99 静默归入"点踩"分支并 {@code setRating(...)} 落库（脏数据）。
 * 与 {@code capability}/{@code groupBy}/{@code dimension} 同属"白名单取值校验保持 service 层单一信息源"裁决。
 */
@DisplayName("AiFeedbackService 评分白名单")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AiFeedbackRatingWhitelistTest {

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

    private SysAiMessage assistantMessage() {
        SysAiMessage message = new SysAiMessage();
        message.setId(MESSAGE_ID);
        message.setConversationId(1L);
        message.setRole("assistant");
        message.setModel("qwen3-0.6b");
        message.setCreateTime(LocalDateTime.now());
        return message;
    }

    private AiFeedbackCreateForm form(int rating) {
        AiFeedbackCreateForm form = new AiFeedbackCreateForm();
        form.setRating(rating);
        return form;
    }

    @ParameterizedTest(name = "rating={0} 不在白名单 → A0400 且不落库")
    @ValueSource(ints = {0, 2, 99, -2})
    @DisplayName("白名单外评分一律 A0400，不得静默归类后落库")
    void ratingOutsideWhitelistRejected(int rating) {
        when(conversationService.getOwnedMessage(MESSAGE_ID, USER_ID)).thenReturn(assistantMessage());

        BusinessException ex = assertThrows(BusinessException.class,
                () -> service.submit(MESSAGE_ID, USER_ID, form(rating)));

        assertThat(ex.getResultCode()).isEqualTo(ResultCode.PARAM_ERROR);
        verifyNoInteractions(feedbackMapper);
    }

    @Test
    @DisplayName("rating=1（点赞）放行并落库，白名单不误伤")
    void likeRatingAccepted() {
        when(conversationService.getOwnedMessage(MESSAGE_ID, USER_ID)).thenReturn(assistantMessage());

        service.submit(MESSAGE_ID, USER_ID, form(1));

        verify(feedbackMapper).insert(any());
    }

    @Test
    @DisplayName("rating=-1（点踩，带问题标签）放行并落库")
    void dislikeRatingAccepted() {
        when(conversationService.getOwnedMessage(MESSAGE_ID, USER_ID)).thenReturn(assistantMessage());
        AiFeedbackCreateForm form = form(-1);
        form.setTags(List.of("incorrect"));

        service.submit(MESSAGE_ID, USER_ID, form);

        verify(feedbackMapper).insert(any());
    }
}
