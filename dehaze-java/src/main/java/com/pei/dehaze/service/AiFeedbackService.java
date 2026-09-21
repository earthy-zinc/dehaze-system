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
import com.pei.dehaze.model.vo.AiFeedbackVO;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * AI 消息反馈服务。
 *
 * <p>对齐 dehaze-python {@code ai_feedback_service}：30 天反馈时效、点赞/点踩标签白名单、
 * (message_id,user_id) 唯一键 upsert（软删行复活）、点踩沉淀用户偏好记忆。
 *
 * @author dehaze
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class AiFeedbackService {

    private static final int FEEDBACK_VALID_DAYS = 30;

    private static final Set<String> LIKE_TAGS = Set.of("accurate", "detailed", "concise", "creative");

    private static final Set<String> DISLIKE_TAGS = Set.of("incorrect", "irrelevant", "incomplete",
            "too_long", "bad_citation", "harmful");

    /** 点踩标签 → 偏好语义记忆（source=feedback，is_preference=1），供记忆注入常驻生效 */
    private static final Map<String, String> DISLIKE_PREFERENCE_MEMORIES = Map.of(
            "too_long", "用户偏好简洁回复",
            "incomplete", "用户期望回复完整、覆盖全部要点",
            "irrelevant", "用户期望回复紧扣主题、避免无关内容");

    private final SysAiMessageFeedbackMapper feedbackMapper;

    private final SysAiMemoryMapper memoryMapper;

    private final AiConversationService conversationService;

    private final ObjectMapper objectMapper;

    @Transactional
    public AiFeedbackVO submit(Long messageId, Long userId, AiFeedbackCreateForm form) {
        SysAiMessage message = conversationService.getOwnedMessage(messageId, userId);
        if (message == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "消息不存在");
        }
        if (!"assistant".equals(message.getRole())) {
            throw new BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "仅助手消息可反馈");
        }
        if (message.getCreateTime() != null
                && message.getCreateTime().isBefore(LocalDateTime.now().minusDays(FEEDBACK_VALID_DAYS))) {
            throw new BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "已超过反馈时效(30天)");
        }
        // 白名单取值校验留在 service 层（与 capability/groupBy/dimension 同裁决）：python 侧为 Literal[1,-1]，
        // 用 @Min(-1)@Max(1) 会放行 0，且下面按 equals(1) 二分会把 0/99 静默当点踩落库
        if (!Integer.valueOf(1).equals(form.getRating()) && !Integer.valueOf(-1).equals(form.getRating())) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "评分仅支持1(点赞)/-1(点踩)");
        }
        List<String> tags = form.getTags();
        if (Integer.valueOf(1).equals(form.getRating())) {
            if (tags != null && !tags.isEmpty() && !LIKE_TAGS.containsAll(tags)) {
                throw new BusinessException(ResultCode.PARAM_ERROR, "不支持的标签类型");
            }
        } else {
            if (tags == null || tags.isEmpty()) {
                throw new BusinessException(ResultCode.PARAM_ERROR, "点踩必须选择问题标签");
            }
            if (!DISLIKE_TAGS.containsAll(tags)) {
                throw new BusinessException(ResultCode.PARAM_ERROR, "不支持的标签类型");
            }
        }
        SysAiMessageFeedback feedback = feedbackMapper.selectByUserAndMessageIgnoringDeleted(messageId, userId);
        if (feedback != null) {
            feedbackMapper.revive(feedback.getId(), form.getRating(), writeTags(tags), form.getComment(),
                    message.getConversationId(), message.getModel(), "internal");
            feedback = feedbackMapper.selectById(feedback.getId());
        } else {
            feedback = new SysAiMessageFeedback();
            feedback.setMessageId(messageId);
            feedback.setUserId(userId);
            feedback.setConversationId(message.getConversationId());
            feedback.setModel(message.getModel());
            feedback.setSource("internal");
            feedback.setRating(form.getRating());
            feedback.setTags(tags);
            feedback.setComment(form.getComment());
            feedback.setProcessed(0);
            feedbackMapper.insert(feedback);
        }
        if (!Integer.valueOf(1).equals(form.getRating())) {
            savePreferenceMemory(userId, tags, form.getComment());
        }
        return toVO(feedback, tags);
    }

    public AiFeedbackVO get(Long messageId, Long userId) {
        SysAiMessageFeedback feedback = feedbackMapper.selectByUserAndMessageIgnoringDeleted(messageId, userId);
        if (feedback == null || (feedback.getDeleted() != null && feedback.getDeleted() != 0)) {
            return null;
        }
        return toVO(feedback, feedback.getTags() == null ? List.of() : feedback.getTags());
    }

    @Transactional
    public void revoke(Long messageId, Long userId) {
        if (get(messageId, userId) == null) {
            throw new BusinessException(ResultCode.FEEDBACK_NOT_FOUND, "反馈不存在");
        }
        feedbackMapper.softDeleteByUserAndMessage(messageId, userId);
    }

    /**
     * 点踩标签沉淀为用户偏好语义记忆（与反馈同事务提交，避免"已反馈但无偏好"的静默缺口）
     */
    private void savePreferenceMemory(Long userId, List<String> tags, String comment) {
        String mapped = null;
        for (String tag : tags) {
            if (DISLIKE_PREFERENCE_MEMORIES.containsKey(tag)) {
                mapped = DISLIKE_PREFERENCE_MEMORIES.get(tag);
                break;
            }
        }
        if (mapped == null) {
            return;
        }
        SysAiMemory memory = new SysAiMemory();
        memory.setUserId(userId);
        memory.setMemoryType("semantic");
        memory.setContent(comment != null && !comment.isBlank()
                ? mapped + "（用户补充：" + comment.trim() + "）" : mapped);
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("category", "preference");
        metadata.put("is_preference", 1);
        memory.setMetadata(metadata);
        memory.setImportance(100);
        memory.setSource("feedback");
        memory.setStatus(1);
        memory.setArchived(0);
        memory.setAccessCount(0);
        memoryMapper.insert(memory);
    }

    private String writeTags(List<String> tags) {
        if (tags == null) {
            return null;
        }
        try {
            return objectMapper.writeValueAsString(tags);
        } catch (Exception e) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "标签格式非法");
        }
    }

    private AiFeedbackVO toVO(SysAiMessageFeedback feedback, List<String> tags) {
        AiFeedbackVO vo = new AiFeedbackVO();
        vo.setId(feedback.getId());
        vo.setMessageId(feedback.getMessageId());
        vo.setUserId(feedback.getUserId());
        vo.setRating(feedback.getRating());
        vo.setTags(tags);
        vo.setComment(feedback.getComment());
        vo.setCreateTime(feedback.getCreateTime());
        vo.setUpdateTime(feedback.getUpdateTime());
        return vo;
    }
}
