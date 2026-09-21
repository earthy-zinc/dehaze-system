package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.AiAgentCopyForm;
import com.pei.dehaze.model.form.AiAgentCreateForm;
import com.pei.dehaze.model.form.AiAgentMcpForm;
import com.pei.dehaze.model.form.AiAgentSkillsForm;
import com.pei.dehaze.model.form.AiAgentStatusForm;
import com.pei.dehaze.model.form.AiAgentSubAgentsForm;
import com.pei.dehaze.model.form.AiAgentUpdateForm;
import com.pei.dehaze.model.query.AiAgentPageQuery;
import com.pei.dehaze.model.query.PageParamQuery;
import com.pei.dehaze.model.vo.AiAgentConfigDefaultsVO;
import com.pei.dehaze.model.vo.AiAgentVO;
import com.pei.dehaze.model.vo.AiAgentVersionVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.AiAgentConfigResolver;
import com.pei.dehaze.service.AiAgentService;
import com.pei.dehaze.service.AiAgentVersionService;
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
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;

/**
 * AI 智能体管理（CRUD/启停/复制/关联绑定/版本查询与回滚）。
 *
 * <p>依赖 LLM 判分的端点不在本控制器：试运行（POST /{id}/test）、发布（POST /{id}/publish，
 * 发布前须真实执行回归评测）与评测执行，均登记在 {@code AiProxyRoutes} 白名单转发 dehaze-python
 * （行为事实源），本控制器不再声明这些路径。
 *
 * @author dehaze
 */
@Tag(name = "29.AI对话")
@RestController
@RequestMapping("/api/v1/ai/agents")
@RequiredArgsConstructor
public class AiAgentController {

    private static final String MANAGE_PERMISSION = "ai:agent:manage";

    private final AiAgentService agentService;

    private final AiAgentVersionService agentVersionService;

    private final AiAgentConfigResolver agentConfigResolver;

    @Operation(summary = "Agent 列表（管理员为分页全量，普通用户为可选列表）")
    @GetMapping
    public PageResult<AiAgentVO> list(@Valid @ParameterObject AiAgentPageQuery query) {
        if (SecurityUtils.isRoot() || SecurityUtils.getPerms().contains(MANAGE_PERMISSION)) {
            return PageResult.success(agentService.list(query));
        }
        List<AiAgentVO> items = agentService.listEnabled();
        Page<AiAgentVO> page = new Page<>(query.getPageNum(), items.size(), items.size());
        page.setRecords(items);
        return PageResult.success(page);
    }

    @Operation(summary = "可选 Agent 列表")
    @GetMapping("/enabled")
    public Result<List<AiAgentVO>> listEnabled() {
        return Result.success(agentService.listEnabled());
    }

    @Operation(summary = "推理参数系统默认值（Agent 配置表单空值继承提示）")
    @GetMapping("/config-defaults")
    public Result<AiAgentConfigDefaultsVO> configDefaults() {
        return Result.success(agentConfigResolver.reasoningDefaults());
    }

    @Operation(summary = "创建 Agent")
    @PostMapping
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public Result<AiAgentVO> create(@Valid @RequestBody AiAgentCreateForm form) {
        return Result.success(agentService.create(form));
    }

    @Operation(summary = "Agent 详情")
    @GetMapping("/{agentId}")
    public Result<AiAgentVO> detail(@Parameter(description = "Agent ID") @PathVariable Long agentId) {
        return Result.success(agentService.getDetail(agentId));
    }

    @Operation(summary = "更新 Agent")
    @PutMapping("/{agentId}")
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public Result<AiAgentVO> update(@Parameter(description = "Agent ID") @PathVariable Long agentId,
                                    @Valid @RequestBody AiAgentUpdateForm form) {
        return Result.success(agentService.update(agentId, form));
    }

    @Operation(summary = "删除 Agent（级联清理评测资产）")
    @DeleteMapping("/{agentId}")
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public Result<Void> delete(@Parameter(description = "Agent ID") @PathVariable Long agentId) {
        agentService.delete(agentId);
        return Result.success();
    }

    @Operation(summary = "启停 Agent")
    @PatchMapping("/{agentId}/status")
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public Result<Void> setStatus(@Parameter(description = "Agent ID") @PathVariable Long agentId,
                                  @Valid @RequestBody AiAgentStatusForm form) {
        agentService.setStatus(agentId, form.getStatus());
        return Result.success();
    }

    @Operation(summary = "复制 Agent")
    @PostMapping("/{agentId}/copy")
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public Result<AiAgentVO> copy(@Parameter(description = "Agent ID") @PathVariable Long agentId,
                                  @Valid @RequestBody AiAgentCopyForm form) {
        return Result.success(agentService.copy(agentId, form.getAgentCode()));
    }

    @Operation(summary = "设置 Skills（覆盖式）")
    @PutMapping("/{agentId}/skills")
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public Result<Void> setSkills(@Parameter(description = "Agent ID") @PathVariable Long agentId,
                                  @Valid @RequestBody AiAgentSkillsForm form) {
        agentService.setSkills(agentId, form.getSkills());
        return Result.success();
    }

    @Operation(summary = "设置 MCP 命名空间（覆盖式）")
    @PutMapping("/{agentId}/mcps")
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public Result<Void> setMcps(@Parameter(description = "Agent ID") @PathVariable Long agentId,
                                @Valid @RequestBody AiAgentMcpForm form) {
        agentService.setMcpNamespaces(agentId, form.getMcpNamespaces());
        return Result.success();
    }

    @Operation(summary = "设置子 Agent（覆盖式）")
    @PutMapping("/{agentId}/subagents")
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public Result<Void> setSubagents(@Parameter(description = "Agent ID") @PathVariable Long agentId,
                                     @Valid @RequestBody AiAgentSubAgentsForm form) {
        agentService.setSubagents(agentId, form);
        return Result.success();
    }

    @Operation(summary = "版本历史")
    @GetMapping("/{agentId}/versions")
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public PageResult<AiAgentVersionVO> versions(
            @Parameter(description = "Agent ID") @PathVariable Long agentId,
            @Valid @ParameterObject PageParamQuery query) {
        return PageResult.success(agentVersionService.listVersions(agentId,
                query.getPageNum(), query.getPageSize()));
    }

    @Operation(summary = "版本差异对比")
    @GetMapping("/{agentId}/versions/diff")
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public Result<List<Map<String, Object>>> diffVersions(
            @Parameter(description = "Agent ID") @PathVariable Long agentId,
            @Parameter(description = "基准版本号") @RequestParam Integer base,
            @Parameter(description = "目标版本号") @RequestParam Integer target) {
        return Result.success(agentVersionService.diffVersions(agentId, base, target));
    }

    @Operation(summary = "版本快照详情")
    @GetMapping("/{agentId}/versions/{versionNo}")
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public Result<AiAgentVersionVO> versionDetail(
            @Parameter(description = "Agent ID") @PathVariable Long agentId,
            @Parameter(description = "版本号") @PathVariable Integer versionNo) {
        return Result.success(agentVersionService.getVersionDetail(agentId, versionNo));
    }

    @Operation(summary = "回滚到历史已发布版本")
    @PostMapping("/{agentId}/versions/{versionNo}/rollback")
    @PreAuthorize("@ss.hasPerm('ai:agent:manage')")
    public Result<Map<String, Object>> rollback(
            @Parameter(description = "Agent ID") @PathVariable Long agentId,
            @Parameter(description = "版本号") @PathVariable Integer versionNo) {
        int newVersionNo = agentVersionService.rollback(agentId, versionNo, SecurityUtils.getUserId());
        return Result.success(Map.of("version_no", newVersionNo));
    }
}
