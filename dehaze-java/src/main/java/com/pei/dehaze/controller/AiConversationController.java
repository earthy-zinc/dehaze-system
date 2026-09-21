package com.pei.dehaze.controller;

import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.AiConversationBatchForm;
import com.pei.dehaze.model.form.AiConversationCreateForm;
import com.pei.dehaze.model.form.AiConversationUpdateForm;
import com.pei.dehaze.model.query.AiConversationPageQuery;
import com.pei.dehaze.model.query.AiMessageCursorQuery;
import com.pei.dehaze.model.query.PageParamQuery;
import com.pei.dehaze.model.vo.AiConversationVO;
import com.pei.dehaze.model.vo.AiMessagePageVO;
import com.pei.dehaze.model.vo.AiMessageVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.AiConversationService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * AI 会话与消息（非推理类端点）。
 *
 * <p>send/regenerate/resume/edit/stop/SSE 重连属转发域，不在本控制器。
 *
 * @author dehaze
 */
@Tag(name = "29.AI对话")
@RestController
@RequestMapping("/api/v1/ai")
@RequiredArgsConstructor
public class AiConversationController {

    private final AiConversationService conversationService;

    @Operation(summary = "创建会话")
    @PostMapping("/conversations")
    public Result<AiConversationVO> create(@Valid @RequestBody AiConversationCreateForm form) {
        return Result.success(conversationService.create(currentUserId(), form));
    }

    @Operation(summary = "会话列表（view=admin 为管理端会话审计，需 ai:conversation:audit）")
    @GetMapping("/conversations")
    public PageResult<AiConversationVO> list(@Valid @ParameterObject AiConversationPageQuery query) {
        return PageResult.success(conversationService.list(currentUserId(), query, isAdminView(query.getView())));
    }

    @Operation(summary = "回收站列表")
    @GetMapping("/conversations/trash")
    public PageResult<AiConversationVO> listTrash(@Valid @ParameterObject PageParamQuery query) {
        return PageResult.success(conversationService.listTrash(currentUserId(),
                query.getPageNum(), query.getPageSize()));
    }

    @Operation(summary = "批量操作会话(archive/restore/delete)")
    @PostMapping("/conversations/batch")
    public Result<Integer> batch(@Valid @RequestBody AiConversationBatchForm form) {
        return Result.success(conversationService.batchOperate(currentUserId(), form));
    }

    @Operation(summary = "会话详情")
    @GetMapping("/conversations/{convId}")
    public Result<AiConversationVO> detail(@Parameter(description = "会话ID") @PathVariable Long convId,
                                          @Parameter(description = "视角(admin)") @RequestParam(required = false) String view) {
        return Result.success(conversationService.getDetail(convId, currentUserId(), isAdminView(view)));
    }

    @Operation(summary = "更新会话")
    @PatchMapping("/conversations/{convId}")
    public Result<AiConversationVO> update(@Parameter(description = "会话ID") @PathVariable Long convId,
                                           @Valid @RequestBody AiConversationUpdateForm form) {
        return Result.success(conversationService.update(convId, currentUserId(), form));
    }

    @Operation(summary = "删除会话（软删，30 天内可恢复）")
    @DeleteMapping("/conversations/{convId}")
    public Result<Void> delete(@Parameter(description = "会话ID") @PathVariable Long convId) {
        conversationService.delete(convId, currentUserId());
        return Result.success();
    }

    @Operation(summary = "恢复软删会话")
    @PostMapping("/conversations/{convId}/restore")
    public Result<AiConversationVO> restore(@Parameter(description = "会话ID") @PathVariable Long convId) {
        return Result.success(conversationService.restore(convId, currentUserId()));
    }

    @Operation(summary = "置顶会话")
    @PutMapping("/conversations/{convId}/pin")
    public Result<AiConversationVO> pin(@Parameter(description = "会话ID") @PathVariable Long convId) {
        return Result.success(conversationService.pin(convId, currentUserId()));
    }

    @Operation(summary = "取消置顶")
    @PutMapping("/conversations/{convId}/unpin")
    public Result<AiConversationVO> unpin(@Parameter(description = "会话ID") @PathVariable Long convId) {
        return Result.success(conversationService.unpin(convId, currentUserId()));
    }

    @Operation(summary = "标记会话已读")
    @PutMapping("/conversations/{convId}/read")
    public Result<AiConversationVO> markRead(@Parameter(description = "会话ID") @PathVariable Long convId) {
        return Result.success(conversationService.markRead(convId, currentUserId()));
    }

    @Operation(summary = "导出会话（markdown/json）")
    @GetMapping("/conversations/{convId}/export")
    public void export(@Parameter(description = "会话ID") @PathVariable Long convId,
                       @Parameter(description = "导出格式") @RequestParam(defaultValue = "markdown") String format,
                       HttpServletResponse response) throws IOException {
        AiConversationService.ConversationExport export =
                conversationService.export(convId, currentUserId(), format);
        response.setContentType(export.contentType());
        response.setCharacterEncoding(StandardCharsets.UTF_8.name());
        response.setHeader("Content-Disposition", "attachment; filename=\""
                + URLEncoder.encode(export.filename(), StandardCharsets.UTF_8) + "\"");
        response.getWriter().write(export.content());
    }

    @Operation(summary = "会话消息列表（游标分页，id 倒序；assistant 消息附带推理步骤）")
    @GetMapping("/conversations/{convId}/messages")
    public Result<AiMessagePageVO> listMessages(
            @Parameter(description = "会话ID") @PathVariable Long convId,
            @Valid @ParameterObject AiMessageCursorQuery query,
            @Parameter(description = "视角(admin)") @RequestParam(required = false) String view) {
        return Result.success(conversationService.listMessages(convId, currentUserId(),
                query.getBefore(), query.getLimit(), isAdminView(view)));
    }

    @Operation(summary = "消息详情（含推理步骤）")
    @GetMapping("/messages/{msgId}")
    public Result<AiMessageVO> getMessage(@Parameter(description = "消息ID") @PathVariable Long msgId,
                                          @Parameter(description = "视角(admin)") @RequestParam(required = false) String view) {
        return Result.success(conversationService.getMessage(msgId, currentUserId(), isAdminView(view)));
    }

    @Operation(summary = "删除助手回复消息（软删除）")
    @DeleteMapping("/messages/{msgId}")
    public Result<Void> deleteMessage(@Parameter(description = "消息ID") @PathVariable Long msgId) {
        conversationService.deleteMessage(msgId, currentUserId());
        return Result.success();
    }

    @Operation(summary = "查询消息的分支列表")
    @GetMapping("/conversations/{convId}/messages/{msgId}/branches")
    public Result<List<AiMessageVO>> branches(@Parameter(description = "会话ID") @PathVariable Long convId,
                                              @Parameter(description = "消息ID") @PathVariable Long msgId) {
        return Result.success(conversationService.getBranches(convId, currentUserId(), msgId));
    }

    @Operation(summary = "切换当前分支")
    @PutMapping("/conversations/{convId}/branches/{msgId}")
    public Result<AiConversationVO> switchBranch(@Parameter(description = "会话ID") @PathVariable Long convId,
                                                 @Parameter(description = "分支末端消息ID") @PathVariable Long msgId) {
        return Result.success(conversationService.switchBranch(convId, currentUserId(), msgId));
    }

    private Long currentUserId() {
        return SecurityUtils.getUserId();
    }

    /**
     * 管理端会话审计视角：ROOT 放行，否则需 ai:conversation:audit；越权抛 AccessDeniedException
     * （由 GlobalExceptionHandler 统一返回 403 + A0301，与 python 的 403 口径一致）
     */
    private boolean isAdminView(String view) {
        if (!"admin".equals(view)) {
            return false;
        }
        if (SecurityUtils.isRoot() || SecurityUtils.getPerms().contains("ai:conversation:audit")) {
            return true;
        }
        throw new AccessDeniedException("会话审计视角越权访问");
    }
}
