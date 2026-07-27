package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.MessageSendForm;
import com.pei.dehaze.model.query.MessageQuery;
import com.pei.dehaze.model.query.MessageSearchQuery;
import com.pei.dehaze.model.vo.*;
import com.pei.dehaze.service.MessageService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.web.bind.annotation.*;

@Tag(name = "10.消息通知")
@RestController
@RequestMapping("/api/v1/messages")
@RequiredArgsConstructor
public class MessageController {

    private final MessageService messageService;

    @Operation(summary = "内部消息发送")
    @PostMapping("/send")
    public Result<MessageSendResultVO> send(@Valid @RequestBody MessageSendForm form) {
        return Result.success(messageService.send(form));
    }

    @Operation(summary = "消息列表（分页）")
    @GetMapping
    public PageResult<MessageVO> getPage(@ParameterObject MessageQuery query) {
        Page<MessageVO> page = messageService.getPage(query);
        return PageResult.success(page);
    }

    @Operation(summary = "未读消息数")
    @GetMapping("/unread-count")
    public Result<UnreadCountVO> getUnreadCount() {
        return Result.success(messageService.getUnreadCount());
    }

    @Operation(summary = "搜索消息")
    @GetMapping("/search")
    public PageResult<MessageVO> search(@Valid @ParameterObject MessageSearchQuery query) {
        Page<MessageVO> page = messageService.search(query);
        return PageResult.success(page);
    }

    @Operation(summary = "消息详情（自动标记已读）")
    @GetMapping("/{id}")
    public Result<MessageDetailVO> getDetail(@Parameter(description = "消息ID") @PathVariable Long id) {
        return Result.success(messageService.getDetail(id));
    }

    @Operation(summary = "标记单条已读")
    @PutMapping("/{id}/read")
    public Result<Void> markRead(@Parameter(description = "消息ID") @PathVariable Long id) {
        messageService.markRead(id);
        return Result.success();
    }

    @Operation(summary = "全部标记已读")
    @PutMapping("/read-all")
    public Result<ReadAllResultVO> markAllRead(@RequestParam(required = false) String type) {
        return Result.success(messageService.markAllRead(type));
    }

    @Operation(summary = "删除消息（支持批量）")
    @DeleteMapping("/{ids}")
    public Result<Void> deleteByIds(
            @Parameter(description = "消息ID，多个用逗号分隔") @PathVariable String ids) {
        messageService.deleteByIds(ids);
        return Result.success();
    }
}
