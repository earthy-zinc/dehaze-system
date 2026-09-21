package com.pei.dehaze.controller;

import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.query.PageParamQuery;
import com.pei.dehaze.model.vo.AiArtifactVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.AiArtifactService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;

/**
 * AI 中间产物（引用化查询）。
 *
 * @author dehaze
 */
@Tag(name = "29.AI对话")
@RestController
@RequestMapping("/api/v1/ai")
@RequiredArgsConstructor
public class AiArtifactController {

    private final AiArtifactService artifactService;

    @Operation(summary = "会话产物分页列表")
    @GetMapping("/conversations/{convId}/artifacts")
    public PageResult<AiArtifactVO> listByConversation(
            @Parameter(description = "会话ID") @PathVariable Long convId,
            @Valid @ParameterObject PageParamQuery query) {
        return PageResult.success(artifactService.listByConversation(convId, SecurityUtils.getUserId(),
                query.getPageNum(), query.getPageSize()));
    }

    @Operation(summary = "消息关联产物列表")
    @GetMapping("/messages/{msgId}/artifacts")
    public Result<List<AiArtifactVO>> listByMessage(@Parameter(description = "消息ID") @PathVariable Long msgId) {
        return Result.success(artifactService.listByMessage(msgId, SecurityUtils.getUserId()));
    }

    @Operation(summary = "按业务引用反查产物列表")
    @GetMapping("/artifacts/by-ref")
    public Result<List<AiArtifactVO>> listByRef(@Parameter(description = "引用业务表") @RequestParam String refType,
                                                @Parameter(description = "引用业务表ID") @RequestParam Long refId) {
        return Result.success(artifactService.listByRef(refType, refId, SecurityUtils.getUserId()));
    }

    @Operation(summary = "产物详情（含运行时图片URL）")
    @GetMapping("/artifacts/{artifactId}/detail")
    public Result<Map<String, Object>> detail(@Parameter(description = "产物ID") @PathVariable Long artifactId) {
        return Result.success(artifactService.getDetail(artifactId, SecurityUtils.getUserId()));
    }
}
