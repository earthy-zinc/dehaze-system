package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.AnnouncementForm;
import com.pei.dehaze.model.query.AnnouncementQuery;
import com.pei.dehaze.model.vo.AnnouncementDetailVO;
import com.pei.dehaze.model.vo.AnnouncementSendResultVO;
import com.pei.dehaze.model.vo.AnnouncementVO;
import com.pei.dehaze.model.vo.IdVO;
import com.pei.dehaze.service.AnnouncementService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

@Tag(name = "10.消息通知-公告管理")
@RestController
@RequestMapping("/api/v1/announcements")
@RequiredArgsConstructor
public class AnnouncementController {

    private final AnnouncementService announcementService;

    @Operation(summary = "公告分页列表")
    @GetMapping("/page")
    public PageResult<AnnouncementVO> getPage(@ParameterObject AnnouncementQuery query) {
        Page<AnnouncementVO> page = announcementService.getPage(query);
        return PageResult.success(page);
    }

    @Operation(summary = "创建公告")
    @PostMapping
    @PreAuthorize("@ss.hasPerm('notify:announcement:add')")
    public Result<IdVO> create(@Valid @RequestBody AnnouncementForm form) {
        Long id = announcementService.create(form);
        return Result.success(new IdVO(id));
    }

    @Operation(summary = "公告详情")
    @GetMapping("/{id}")
    public Result<AnnouncementDetailVO> getDetail(@Parameter(description = "公告ID") @PathVariable Long id) {
        return Result.success(announcementService.getDetail(id));
    }

    @Operation(summary = "编辑公告（仅草稿/待发送）")
    @PutMapping("/{id}")
    @PreAuthorize("@ss.hasPerm('notify:announcement:edit')")
    public Result<Void> update(@Parameter(description = "公告ID") @PathVariable Long id,
                               @RequestBody AnnouncementForm form) {
        announcementService.update(id, form);
        return Result.success();
    }

    @Operation(summary = "删除公告")
    @DeleteMapping("/{id}")
    @PreAuthorize("@ss.hasPerm('notify:announcement:delete')")
    public Result<Void> delete(@Parameter(description = "公告ID") @PathVariable Long id) {
        announcementService.delete(id);
        return Result.success();
    }

    @Operation(summary = "立即发送公告")
    @PostMapping("/{id}/send")
    @PreAuthorize("@ss.hasPerm('notify:announcement:send')")
    public Result<AnnouncementSendResultVO> send(@Parameter(description = "公告ID") @PathVariable Long id) {
        return Result.success(announcementService.send(id));
    }

    @Operation(summary = "取消定时公告")
    @PutMapping("/{id}/cancel")
    @PreAuthorize("@ss.hasPerm('notify:announcement:cancel')")
    public Result<Void> cancel(@Parameter(description = "公告ID") @PathVariable Long id) {
        announcementService.cancel(id);
        return Result.success();
    }
}
