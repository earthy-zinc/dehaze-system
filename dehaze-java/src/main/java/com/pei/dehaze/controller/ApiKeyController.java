package com.pei.dehaze.controller;

import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.dto.ApiKeyResult;
import com.pei.dehaze.model.form.ApiKeyForm;
import com.pei.dehaze.service.ApiKeyService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Tag(name = "01.认证中心")
@RestController
@RequestMapping("/api/v1/auth/api-keys")
@RequiredArgsConstructor
public class ApiKeyController {

    private final ApiKeyService apiKeyService;

    @Operation(summary = "创建API密钥")
    @PostMapping("")
    public Result<ApiKeyResult> createApiKey(@Valid @RequestBody ApiKeyForm form) {
        return Result.success(apiKeyService.createApiKey(form));
    }

    @Operation(summary = "获取API密钥列表(仅未吊销)")
    @GetMapping("")
    public Result<List<ApiKeyResult>> listApiKeys() {
        return Result.success(apiKeyService.listApiKeys());
    }

    @Operation(summary = "吊销API密钥(内部设 revoked_at=now(), 不再物理删除)")
    @DeleteMapping("/{id}")
    public Result<Void> deleteApiKey(@PathVariable Long id) {
        return Result.judge(apiKeyService.revokeApiKey(id));
    }
}
