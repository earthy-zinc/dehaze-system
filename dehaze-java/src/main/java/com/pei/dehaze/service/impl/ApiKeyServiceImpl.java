package com.pei.dehaze.service.impl;

import cn.hutool.core.util.RandomUtil;
import cn.hutool.crypto.SecureUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.mapper.SysAiModelMapper;
import com.pei.dehaze.mapper.SysApiKeyMapper;
import com.pei.dehaze.model.dto.ApiKeyResult;
import com.pei.dehaze.model.dto.UserAuthInfo;
import com.pei.dehaze.model.entity.SysAiModel;
import com.pei.dehaze.model.entity.SysApiKey;
import com.pei.dehaze.model.form.ApiKeyForm;
import com.pei.dehaze.security.model.SysUserDetails;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.ApiKeyService;
import com.pei.dehaze.service.SysUserService;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import lombok.RequiredArgsConstructor;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class ApiKeyServiceImpl extends ServiceImpl<SysApiKeyMapper, SysApiKey> implements ApiKeyService {

    private final SysUserService sysUserService;
    private final SysApiKeyMapper sysApiKeyMapper;
    private final SysAiModelMapper sysAiModelMapper;

    @Override
    public ApiKeyResult createApiKey(ApiKeyForm form) {
        validateModelWhitelist(form.getModelWhitelist());
        String rawKey = "dhak_" + RandomUtil.randomString(48);
        String keyHash = SecureUtil.sha256(rawKey);
        String prefix = rawKey.substring(0, 9);

        SysApiKey apiKey = new SysApiKey();
        apiKey.setUserId(SecurityUtils.getUserId());
        apiKey.setName(form.getName());
        apiKey.setKeyPrefix(prefix);
        apiKey.setKeyHash(keyHash);
        apiKey.setStatus(1);
        apiKey.setExpiresAt(form.getExpiresAt());
        apiKey.setDailyQuota(form.getDailyQuota());
        apiKey.setMonthlyQuota(form.getMonthlyQuota());
        apiKey.setRpmLimit(form.getRpmLimit());
        // 空数组与不传同义（python `model_whitelist or None`）：落 NULL 而非 JSON 空数组
        List<String> whitelist = form.getModelWhitelist();
        apiKey.setModelWhitelist(whitelist == null || whitelist.isEmpty() ? null : whitelist);
        this.save(apiKey);

        return ApiKeyResult.builder()
                .id(apiKey.getId())
                .name(apiKey.getName())
                .apiKey(rawKey)
                .keyPrefix(apiKey.getKeyPrefix())
                .status(apiKey.getStatus())
                .expiresAt(apiKey.getExpiresAt())
                .dailyQuota(apiKey.getDailyQuota())
                .monthlyQuota(apiKey.getMonthlyQuota())
                .rpmLimit(apiKey.getRpmLimit())
                .modelWhitelist(apiKey.getModelWhitelist())
                .lastUsedAt(apiKey.getLastUsedAt())
                .createTime(apiKey.getCreateTime())
                .build();
    }

    @Override
    public List<ApiKeyResult> listApiKeys() {
        Long userId = SecurityUtils.getUserId();
        // 列表查询排除已吊销的密钥（revoked_at IS NOT NULL）
        List<SysApiKey> keys = this.list(new LambdaQueryWrapper<SysApiKey>()
                .eq(SysApiKey::getUserId, userId)
                .isNull(SysApiKey::getRevokedAt)
                .orderByDesc(SysApiKey::getCreateTime));
        return keys.stream().map(k -> ApiKeyResult.builder()
                .id(k.getId())
                .name(k.getName())
                .keyPrefix(k.getKeyPrefix())
                .status(k.getStatus())
                .expiresAt(k.getExpiresAt())
                .lastUsedAt(k.getLastUsedAt())
                .createTime(k.getCreateTime())
                .dailyQuota(k.getDailyQuota())
                .monthlyQuota(k.getMonthlyQuota())
                .rpmLimit(k.getRpmLimit())
                .modelWhitelist(k.getModelWhitelist())
                .build()).collect(Collectors.toList());
    }

    @Override
    public boolean revokeApiKey(Long id) {
        Long userId = SecurityUtils.getUserId();
        SysApiKey apiKey = this.getById(id);
        if (apiKey == null || !apiKey.getUserId().equals(userId)) {
            // 越权与不存在同口径防枚举（python A0401）
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "API密钥不存在");
        }
        // 吊销：设 revoked_at = now()，不再物理删除、不再写 deleted
        return this.update(new LambdaUpdateWrapper<SysApiKey>()
                .eq(SysApiKey::getId, id)
                .set(SysApiKey::getRevokedAt, LocalDateTime.now()));
    }

    /**
     * 白名单内每个模型标识必须存在且启用（对齐 python {@code ApiKeyService._validate_whitelist}），否则 A0400。
     * <p>
     * 拼错或已停用的模型写进白名单后，该 Key 的兼容调用会全部 403
     * （python {@code compatible_governance._check_whitelist}），调用方无从自查。
     */
    private void validateModelWhitelist(List<String> whitelist) {
        if (whitelist == null || whitelist.isEmpty()) {
            return;
        }
        for (String modelId : whitelist) {
            Long count = sysAiModelMapper.selectCount(new LambdaQueryWrapper<SysAiModel>()
                    .eq(SysAiModel::getModelId, modelId)
                    .eq(SysAiModel::getStatus, 1));
            if (count == null || count == 0) {
                throw new BusinessException(ResultCode.PARAM_ERROR, "模型 " + modelId + " 不存在或未启用");
            }
        }
    }

    @Override
    public Authentication authenticateByKey(String rawKey) {
        String keyHash = SecureUtil.sha256(rawKey);
        // 有效 key 判定：revoked_at IS NULL（而非 deleted=0）
        SysApiKey apiKey = this.getOne(new LambdaQueryWrapper<SysApiKey>()
                .eq(SysApiKey::getKeyHash, keyHash)
                .isNull(SysApiKey::getRevokedAt));
        if (apiKey == null) {
            return null;
        }
        if (apiKey.getStatus() != 1) {
            return null;
        }
        if (apiKey.getExpiresAt() != null && apiKey.getExpiresAt().isBefore(LocalDateTime.now())) {
            return null;
        }

        UserAuthInfo userAuthInfo = sysUserService.getUserAuthInfo(
                sysUserService.getById(apiKey.getUserId()).getUsername());
        if (userAuthInfo == null) {
            return null;
        }

        SysUserDetails userDetails = new SysUserDetails(userAuthInfo);
        if (!userDetails.isEnabled()) {
            return null;
        }

        apiKey.setLastUsedAt(LocalDateTime.now());
        this.updateById(apiKey);

        return new UsernamePasswordAuthenticationToken(userDetails, "", userDetails.getAuthorities());
    }
}
