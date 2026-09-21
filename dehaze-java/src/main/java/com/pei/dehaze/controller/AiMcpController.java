package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.McpCredentialForm;
import com.pei.dehaze.model.form.McpNamespaceForm;
import com.pei.dehaze.model.form.McpServerForm;
import com.pei.dehaze.model.form.McpServerStatusForm;
import com.pei.dehaze.model.form.McpServerUpdateForm;
import com.pei.dehaze.model.query.McpCallPageQuery;
import com.pei.dehaze.model.query.McpServerPageQuery;
import com.pei.dehaze.model.vo.McpCallVO;
import com.pei.dehaze.model.vo.McpMarketPresetVO;
import com.pei.dehaze.model.vo.McpNamespaceVO;
import com.pei.dehaze.model.vo.McpServerVO;
import com.pei.dehaze.service.AiMcpService;
import io.swagger.v3.oas.annotations.Operation;
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
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@Tag(name = "MCP Server 管理")
@RestController
@RequestMapping("/api/v1/ai/mcp")
@RequiredArgsConstructor
public class AiMcpController {

    private final AiMcpService aiMcpService;

    @Operation(summary = "MCP Server 列表")
    @GetMapping("/servers")
    @PreAuthorize("@ss.hasPerm('ai:mcp:manage')")
    public PageResult<McpServerVO> listServers(@Valid @ParameterObject McpServerPageQuery query) {
        Page<McpServerVO> page = aiMcpService.listServers(query);
        return PageResult.success(page);
    }

    @Operation(summary = "注册外部 MCP Server")
    @PostMapping("/servers")
    @PreAuthorize("@ss.hasPerm('ai:mcp:manage')")
    public Result<McpServerVO> createServer(@RequestBody @Valid McpServerForm form) {
        return Result.success(aiMcpService.createServer(form));
    }

    @Operation(summary = "Server 详情")
    @GetMapping("/servers/{serverId}")
    @PreAuthorize("@ss.hasPerm('ai:mcp:manage')")
    public Result<McpServerVO> getServer(@PathVariable Long serverId) {
        return Result.success(aiMcpService.getServer(serverId));
    }

    @Operation(summary = "更新 Server")
    @PutMapping("/servers/{serverId}")
    @PreAuthorize("@ss.hasPerm('ai:mcp:manage')")
    public Result<McpServerVO> updateServer(@PathVariable Long serverId,
                                            @RequestBody @Valid McpServerUpdateForm form) {
        return Result.success(aiMcpService.updateServer(serverId, form));
    }

    @Operation(summary = "删除 Server")
    @DeleteMapping("/servers/{serverId}")
    @PreAuthorize("@ss.hasPerm('ai:mcp:manage')")
    public Result<Void> deleteServer(@PathVariable Long serverId) {
        aiMcpService.deleteServer(serverId);
        return Result.success();
    }

    @Operation(summary = "启停 Server")
    @PatchMapping("/servers/{serverId}/status")
    @PreAuthorize("@ss.hasPerm('ai:mcp:manage')")
    public Result<McpServerVO> switchServerStatus(@PathVariable Long serverId,
                                                  @RequestBody @Valid McpServerStatusForm form) {
        return Result.success(aiMcpService.switchServerStatus(serverId, form.getStatus()));
    }

    @Operation(summary = "命名空间列表")
    @GetMapping("/servers/{serverId}/namespaces")
    @PreAuthorize("@ss.hasPerm('ai:mcp:manage')")
    public Result<List<McpNamespaceVO>> listNamespaces(@PathVariable Long serverId) {
        return Result.success(aiMcpService.listNamespaces(serverId));
    }

    @Operation(summary = "配置命名空间")
    @PutMapping("/servers/{serverId}/namespaces")
    @PreAuthorize("@ss.hasPerm('ai:mcp:manage')")
    public Result<List<McpNamespaceVO>> updateNamespaces(@PathVariable Long serverId,
                                                         @RequestBody List<McpNamespaceForm> namespaces) {
        return Result.success(aiMcpService.updateNamespaces(serverId, namespaces));
    }

    @Operation(summary = "配置外部服务凭据")
    @PutMapping("/servers/{serverId}/credentials")
    @PreAuthorize("@ss.hasPerm('ai:mcp:manage')")
    public Result<Void> updateCredentials(@PathVariable Long serverId,
                                          @RequestBody @Valid McpCredentialForm form) {
        aiMcpService.updateCredentials(serverId, form);
        return Result.success();
    }

    @Operation(summary = "MCP 市场目录")
    @GetMapping("/market")
    @PreAuthorize("@ss.hasPerm('ai:mcp:manage')")
    public Result<List<McpMarketPresetVO>> getMarket() {
        return Result.success(aiMcpService.getMarket());
    }

    @Operation(summary = "从市场接入预设 Server")
    @PostMapping("/market/{presetId}/install")
    @PreAuthorize("@ss.hasPerm('ai:mcp:manage')")
    public Result<McpServerVO> installPreset(@PathVariable String presetId) {
        return Result.success(aiMcpService.installPreset(presetId));
    }

    @Operation(summary = "MCP 调用审计")
    @GetMapping("/calls")
    @PreAuthorize("@ss.hasPerm('ai:mcp:manage')")
    public PageResult<McpCallVO> listCalls(@Valid @ParameterObject McpCallPageQuery query) {
        Page<McpCallVO> page = aiMcpService.listCalls(query);
        return PageResult.success(page);
    }
}
