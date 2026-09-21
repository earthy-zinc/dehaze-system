package com.pei.dehaze.controller;

import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.AiEvalDatasetCreateForm;
import com.pei.dehaze.model.form.AiEvalDatasetUpdateForm;
import com.pei.dehaze.model.form.AiEvalSampleCreateForm;
import com.pei.dehaze.model.form.AiEvalSampleUpdateForm;
import com.pei.dehaze.model.query.AiEvalRunPageQuery;
import com.pei.dehaze.model.vo.AiEvalDatasetVO;
import com.pei.dehaze.model.vo.AiEvalRunVO;
import com.pei.dehaze.model.vo.AiEvalSampleVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.AiEvalService;
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
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

/**
 * 智能体评测集/样本/执行记录（评测执行与任务进度属转发域）。
 *
 * @author dehaze
 */
@Tag(name = "29.AI对话")
@RestController
@RequestMapping("/api/v1/ai/agents/{agentId}/eval")
@RequiredArgsConstructor
public class AiAgentEvalController {

    private final AiEvalService evalService;

    @Operation(summary = "创建评测集")
    @PostMapping("/datasets")
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public Result<AiEvalDatasetVO> createDataset(@Parameter(description = "Agent ID") @PathVariable Long agentId,
                                                 @Valid @RequestBody AiEvalDatasetCreateForm form) {
        return Result.success(evalService.createDataset(agentId, form));
    }

    @Operation(summary = "评测集列表")
    @GetMapping("/datasets")
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public Result<List<AiEvalDatasetVO>> listDatasets(@Parameter(description = "Agent ID") @PathVariable Long agentId) {
        return Result.success(evalService.listDatasets(agentId));
    }

    @Operation(summary = "更新评测集")
    @PatchMapping("/datasets/{datasetId}")
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public Result<AiEvalDatasetVO> updateDataset(
            @Parameter(description = "Agent ID") @PathVariable Long agentId,
            @Parameter(description = "评测集ID") @PathVariable Long datasetId,
            @Valid @RequestBody AiEvalDatasetUpdateForm form) {
        return Result.success(evalService.updateDataset(agentId, datasetId, form));
    }

    @Operation(summary = "删除评测集（级联清理样本）")
    @DeleteMapping("/datasets/{datasetId}")
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public Result<Void> deleteDataset(
            @Parameter(description = "Agent ID") @PathVariable Long agentId,
            @Parameter(description = "评测集ID") @PathVariable Long datasetId) {
        evalService.deleteDataset(agentId, datasetId, SecurityUtils.getUserId());
        return Result.success();
    }

    @Operation(summary = "创建评测样本")
    @PostMapping("/datasets/{datasetId}/samples")
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public Result<AiEvalSampleVO> createSample(
            @Parameter(description = "Agent ID") @PathVariable Long agentId,
            @Parameter(description = "评测集ID") @PathVariable Long datasetId,
            @Valid @RequestBody AiEvalSampleCreateForm form) {
        return Result.success(evalService.createSample(agentId, datasetId, form));
    }

    @Operation(summary = "评测样本列表")
    @GetMapping("/datasets/{datasetId}/samples")
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public Result<List<AiEvalSampleVO>> listSamples(
            @Parameter(description = "Agent ID") @PathVariable Long agentId,
            @Parameter(description = "评测集ID") @PathVariable Long datasetId) {
        return Result.success(evalService.listSamples(agentId, datasetId));
    }

    @Operation(summary = "更新评测样本")
    @PatchMapping("/samples/{sampleId}")
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public Result<AiEvalSampleVO> updateSample(
            @Parameter(description = "Agent ID") @PathVariable Long agentId,
            @Parameter(description = "样本ID") @PathVariable Long sampleId,
            @Valid @RequestBody AiEvalSampleUpdateForm form) {
        return Result.success(evalService.updateSample(agentId, sampleId, form));
    }

    @Operation(summary = "删除评测样本")
    @DeleteMapping("/samples/{sampleId}")
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public Result<Void> deleteSample(@Parameter(description = "Agent ID") @PathVariable Long agentId,
                                     @Parameter(description = "样本ID") @PathVariable Long sampleId) {
        evalService.deleteSample(agentId, sampleId, SecurityUtils.getUserId());
        return Result.success();
    }

    @Operation(summary = "评测执行记录")
    @GetMapping("/runs")
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public PageResult<AiEvalRunVO> listRuns(
            @Parameter(description = "Agent ID") @PathVariable Long agentId,
            @Valid @ParameterObject AiEvalRunPageQuery query) {
        return PageResult.success(evalService.listRuns(agentId, query.getPageNum(), query.getPageSize(),
                query.getDatasetId()));
    }
}
