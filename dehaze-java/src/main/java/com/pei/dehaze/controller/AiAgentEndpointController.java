package com.pei.dehaze.controller;

import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.AiEndpointCreateForm;
import com.pei.dehaze.model.form.AiEndpointUpdateForm;
import com.pei.dehaze.model.query.AiEndpointPageQuery;
import com.pei.dehaze.model.vo.AiEndpointVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.AiAgentEndpointService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * 外部 A2A 端点管理（Agent Card 手动刷新属转发域）。
 *
 * @author dehaze
 */
@Tag(name = "29.AI对话")
@RestController
@RequestMapping("/api/v1/ai/a2a/endpoints")
@RequiredArgsConstructor
public class AiAgentEndpointController {

    private final AiAgentEndpointService endpointService;

    @Operation(summary = "注册外部A2A端点")
    @PostMapping
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public Result<AiEndpointVO> create(@Valid @RequestBody AiEndpointCreateForm form) {
        return Result.success(endpointService.create(form));
    }

    @Operation(summary = "更新端点")
    @PatchMapping("/{endpointId}")
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public Result<AiEndpointVO> update(@Parameter(description = "端点ID") @PathVariable Long endpointId,
                                       @Valid @RequestBody AiEndpointUpdateForm form) {
        return Result.success(endpointService.update(endpointId, form));
    }

    @Operation(summary = "删除端点")
    @DeleteMapping("/{endpointId}")
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public Result<Void> delete(@Parameter(description = "端点ID") @PathVariable Long endpointId) {
        endpointService.delete(endpointId, SecurityUtils.getUserId());
        return Result.success();
    }

    @Operation(summary = "端点分页列表")
    @GetMapping
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public PageResult<AiEndpointVO> list(@Valid @ParameterObject AiEndpointPageQuery query) {
        return PageResult.success(endpointService.list(query));
    }
}
