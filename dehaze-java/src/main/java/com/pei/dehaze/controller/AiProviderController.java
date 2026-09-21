package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.ProviderForm;
import com.pei.dehaze.model.form.ProviderKeyForm;
import com.pei.dehaze.model.form.ProviderKeyUpdateForm;
import com.pei.dehaze.model.form.ProviderUpdateForm;
import com.pei.dehaze.model.query.ProviderPageQuery;
import com.pei.dehaze.model.vo.ProviderEnabledVO;
import com.pei.dehaze.model.vo.ProviderKeyVO;
import com.pei.dehaze.model.vo.ProviderVO;
import com.pei.dehaze.service.AiProviderService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@Tag(name = "AI供应商管理")
@RestController
@RequestMapping("/api/v1/ai/providers")
@RequiredArgsConstructor
public class AiProviderController {

    private final AiProviderService aiProviderService;

    @Operation(summary = "供应商分页列表")
    @GetMapping
    @PreAuthorize("@ss.hasPerm('ai:model:manage')")
    public PageResult<ProviderVO> listProviders(@Valid @ParameterObject ProviderPageQuery query) {
        Page<ProviderVO> page = aiProviderService.listProviders(query);
        return PageResult.success(page);
    }

    @Operation(summary = "启用供应商列表(精简视图,不含供应商内部配置)")
    @GetMapping("/enabled")
    public Result<List<ProviderEnabledVO>> listEnabledProviders() {
        return Result.success(aiProviderService.listEnabledProviders());
    }

    @Operation(summary = "新增供应商")
    @PostMapping
    @PreAuthorize("@ss.hasPerm('ai:model:manage')")
    public Result<ProviderVO> createProvider(@RequestBody @Valid ProviderForm form) {
        return Result.success(aiProviderService.createProvider(form));
    }

    @Operation(summary = "更新供应商")
    @PutMapping("/{providerId}")
    @PreAuthorize("@ss.hasPerm('ai:model:manage')")
    public Result<ProviderVO> updateProvider(@PathVariable Long providerId,
                                             @RequestBody @Valid ProviderUpdateForm form) {
        return Result.success(aiProviderService.updateProvider(providerId, form));
    }

    @Operation(summary = "删除供应商")
    @DeleteMapping("/{providerId}")
    @PreAuthorize("@ss.hasPerm('ai:model:manage')")
    public Result<Void> deleteProvider(@PathVariable Long providerId) {
        aiProviderService.deleteProvider(providerId);
        return Result.success();
    }

    @Operation(summary = "供应商API Key列表")
    @GetMapping("/{providerId}/keys")
    @PreAuthorize("@ss.hasPerm('ai:model:manage')")
    public Result<List<ProviderKeyVO>> listKeys(@PathVariable Long providerId) {
        return Result.success(aiProviderService.listKeys(providerId));
    }

    @Operation(summary = "新增API Key")
    @PostMapping("/{providerId}/keys")
    @PreAuthorize("@ss.hasPerm('ai:model:manage')")
    public Result<ProviderKeyVO> createKey(@PathVariable Long providerId,
                                           @RequestBody @Valid ProviderKeyForm form) {
        return Result.success(aiProviderService.createKey(providerId, form));
    }

    @Operation(summary = "更新API Key")
    @PutMapping("/{providerId}/keys/{keyId}")
    @PreAuthorize("@ss.hasPerm('ai:model:manage')")
    public Result<ProviderKeyVO> updateKey(@PathVariable Long providerId,
                                           @PathVariable Long keyId,
                                           @RequestBody @Valid ProviderKeyUpdateForm form) {
        return Result.success(aiProviderService.updateKey(providerId, keyId, form));
    }

    @Operation(summary = "删除API Key")
    @DeleteMapping("/{providerId}/keys/{keyId}")
    @PreAuthorize("@ss.hasPerm('ai:model:manage')")
    public Result<Void> deleteKey(@PathVariable Long providerId,
                                  @PathVariable Long keyId) {
        aiProviderService.deleteKey(providerId, keyId);
        return Result.success();
    }

    @Operation(summary = "手动解除供应商熔断")
    @PostMapping("/{providerId}/circuit/close")
    @PreAuthorize("@ss.hasPerm('ai:model:manage')")
    public Result<Void> closeProviderCircuit(@PathVariable Long providerId) {
        aiProviderService.closeCircuit(providerId);
        return Result.success();
    }
}
