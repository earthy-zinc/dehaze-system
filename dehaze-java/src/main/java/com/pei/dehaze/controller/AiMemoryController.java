package com.pei.dehaze.controller;

import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.AiMemoryCreateForm;
import com.pei.dehaze.model.form.AiMemoryUpdateForm;
import com.pei.dehaze.model.query.AiMemoryPageQuery;
import com.pei.dehaze.model.vo.AiMemoryVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.AiMemoryService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.validation.Valid;
import jakarta.validation.constraints.Min;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
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
import java.time.LocalDateTime;
import java.util.List;

/**
 * AI 长期记忆。
 *
 * <p>类级 {@code @Validated} 支撑 {@code search} 的 limit 下界（方法级校验）；生产由 Boot 的
 * {@code ValidationAutoConfiguration} 注册 {@code MethodValidationPostProcessor} 生效。
 *
 * @author dehaze
 */
@Tag(name = "29.AI对话")
@RestController
@RequestMapping("/api/v1/ai/memories")
@RequiredArgsConstructor
@Validated
public class AiMemoryController {

    private final AiMemoryService memoryService;

    @Operation(summary = "记忆分页列表")
    @GetMapping
    public PageResult<AiMemoryVO> list(@Valid @ParameterObject AiMemoryPageQuery query) {
        return PageResult.success(memoryService.list(SecurityUtils.getUserId(), query.getPageNum(),
                query.getPageSize(), query.getMemoryType(), query.getSource()));
    }

    @Operation(summary = "归档记忆分页列表")
    @GetMapping("/archived")
    public PageResult<AiMemoryVO> listArchived(@Valid @ParameterObject AiMemoryPageQuery query) {
        return PageResult.success(memoryService.listArchived(SecurityUtils.getUserId(), query.getPageNum(),
                query.getPageSize(), query.getMemoryType()));
    }

    @Operation(summary = "创建记忆")
    @PostMapping
    public Result<AiMemoryVO> create(@Valid @RequestBody AiMemoryCreateForm form) {
        return Result.success(memoryService.create(SecurityUtils.getUserId(), form));
    }

    @Operation(summary = "更新记忆")
    @PutMapping("/{memoryId}")
    public Result<AiMemoryVO> update(@Parameter(description = "记忆ID") @PathVariable Long memoryId,
                                     @Valid @RequestBody AiMemoryUpdateForm form) {
        return Result.success(memoryService.update(memoryId, SecurityUtils.getUserId(), form));
    }

    @Operation(summary = "删除记忆")
    @DeleteMapping("/{memoryId}")
    public Result<Void> delete(@Parameter(description = "记忆ID") @PathVariable Long memoryId) {
        memoryService.delete(memoryId, SecurityUtils.getUserId());
        return Result.success();
    }

    @Operation(summary = "取消归档记忆")
    @PostMapping("/{memoryId}/unarchive")
    public Result<AiMemoryVO> unarchive(@Parameter(description = "记忆ID") @PathVariable Long memoryId) {
        return Result.success(memoryService.unarchive(memoryId, SecurityUtils.getUserId()));
    }

    @Operation(summary = "关键词搜索记忆")
    @GetMapping("/search")
    public Result<List<AiMemoryVO>> search(@Parameter(description = "关键词") @RequestParam String keyword,
                                           @Parameter(description = "返回条数(>=1)") @Min(1)
                                           @RequestParam(defaultValue = "5") int limit) {
        return Result.success(memoryService.search(SecurityUtils.getUserId(), keyword, limit));
    }

    @Operation(summary = "批量清空记忆（需二次确认，30 天内可恢复）")
    @PostMapping("/clear")
    public Result<Integer> clear(
            @Parameter(description = "记忆类型(为空则全部)") @RequestParam(required = false) String memoryType,
            @Parameter(description = "时间范围起") @RequestParam(required = false)
            @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") LocalDateTime start,
            @Parameter(description = "时间范围止") @RequestParam(required = false)
            @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") LocalDateTime end,
            @Parameter(description = "二次确认标识") @RequestParam(defaultValue = "false") Boolean confirm) {
        int count = memoryService.batchClear(SecurityUtils.getUserId(), confirm, memoryType, start, end);
        return Result.success(count, "已清空 " + count + " 条记忆（30 天内可恢复）");
    }

    @Operation(summary = "恢复软删记忆")
    @PostMapping("/restore")
    public Result<Integer> restore(
            @Parameter(description = "记忆类型(为空则全部)") @RequestParam(required = false) String memoryType,
            @Parameter(description = "时间范围起") @RequestParam(required = false)
            @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") LocalDateTime start,
            @Parameter(description = "时间范围止") @RequestParam(required = false)
            @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") LocalDateTime end,
            @Parameter(description = "二次确认标识") @RequestParam(defaultValue = "false") Boolean confirm) {
        int count = memoryService.restoreDeleted(SecurityUtils.getUserId(), confirm, memoryType, start, end);
        return Result.success(count, "已恢复 " + count + " 条记忆");
    }

    @Operation(summary = "导出全部记忆(JSON/Markdown)")
    @GetMapping("/export")
    public void export(@Parameter(description = "导出格式(json/markdown)") @RequestParam(defaultValue = "json") String fmt,
                       HttpServletResponse response) throws IOException {
        AiMemoryService.MemoryExport export = memoryService.export(SecurityUtils.getUserId(), fmt);
        response.setContentType(export.contentType());
        response.setCharacterEncoding(StandardCharsets.UTF_8.name());
        response.setHeader("Content-Disposition", "attachment; filename=\""
                + URLEncoder.encode(export.filename(), StandardCharsets.UTF_8) + "\"");
        response.getWriter().write(export.content());
    }
}
