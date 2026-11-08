package com.pei.clarity.controller;

import com.pei.clarity.common.result.Result;
import com.pei.clarity.model.vo.AiFeedbackVO;
import com.pei.clarity.security.util.SecurityUtils;
import com.pei.clarity.service.AiFeedbackService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import java.util.List;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * AI 会话级反馈查询（进入会话时一次拉全，替代逐条消息请求）。
 *
 * @author clarity
 */
@Tag(name = "29.AI对话")
@RestController
@RequestMapping("/api/v1/ai/conversations/{convId}/feedback")
@RequiredArgsConstructor
public class AiConversationFeedbackController {

    private final AiFeedbackService feedbackService;

    @Operation(summary = "查询会话内当前用户的全部反馈（批量）")
    @GetMapping
    public Result<List<AiFeedbackVO>> list(@Parameter(description = "会话ID") @PathVariable Long convId) {
        return Result.success(feedbackService.listByConversation(convId, SecurityUtils.getUserId()));
    }
}
