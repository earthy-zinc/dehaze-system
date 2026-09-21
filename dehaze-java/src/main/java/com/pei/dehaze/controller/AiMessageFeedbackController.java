package com.pei.dehaze.controller;

import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.AiFeedbackCreateForm;
import com.pei.dehaze.model.vo.AiFeedbackVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.AiFeedbackService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * AI 消息反馈。
 *
 * @author dehaze
 */
@Tag(name = "29.AI对话")
@RestController
@RequestMapping("/api/v1/ai/messages/{messageId}/feedback")
@RequiredArgsConstructor
public class AiMessageFeedbackController {

    private final AiFeedbackService feedbackService;

    @Operation(summary = "提交/更新消息反馈")
    @PostMapping
    public Result<AiFeedbackVO> submit(@Parameter(description = "消息ID") @PathVariable Long messageId,
                                       @Valid @RequestBody AiFeedbackCreateForm form) {
        return Result.success(feedbackService.submit(messageId, SecurityUtils.getUserId(), form));
    }

    @Operation(summary = "查询消息反馈状态")
    @GetMapping
    public Result<AiFeedbackVO> get(@Parameter(description = "消息ID") @PathVariable Long messageId) {
        return Result.success(feedbackService.get(messageId, SecurityUtils.getUserId()));
    }

    @Operation(summary = "撤销消息反馈")
    @DeleteMapping
    public Result<Void> revoke(@Parameter(description = "消息ID") @PathVariable Long messageId) {
        feedbackService.revoke(messageId, SecurityUtils.getUserId());
        return Result.success();
    }
}
