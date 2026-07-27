package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.MessageTemplateForm;
import com.pei.dehaze.model.query.MessageTemplateQuery;
import com.pei.dehaze.model.vo.MessageTemplateDetailVO;
import com.pei.dehaze.model.vo.MessageTemplateVO;
import com.pei.dehaze.service.MessageTemplateService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

@Tag(name = "10.消息通知-模板管理")
@RestController
@RequestMapping("/api/v1/message-templates")
@RequiredArgsConstructor
public class MessageTemplateController {

    private final MessageTemplateService messageTemplateService;

    @Operation(summary = "消息模板分页列表")
    @GetMapping("/page")
    public PageResult<MessageTemplateVO> getPage(@ParameterObject MessageTemplateQuery query) {
        Page<MessageTemplateVO> page = messageTemplateService.getPage(query);
        return PageResult.success(page);
    }

    @Operation(summary = "消息模板详情")
    @GetMapping("/{id}")
    public Result<MessageTemplateDetailVO> getDetail(@Parameter(description = "模板ID") @PathVariable Long id) {
        return Result.success(messageTemplateService.getDetail(id));
    }

    @Operation(summary = "编辑消息模板")
    @PutMapping("/{id}")
    @PreAuthorize("@ss.hasPerm('notify:template:edit')")
    public Result<Void> update(@Parameter(description = "模板ID") @PathVariable Long id,
                               @Valid @RequestBody MessageTemplateForm form) {
        messageTemplateService.update(id, form);
        return Result.success();
    }
}
