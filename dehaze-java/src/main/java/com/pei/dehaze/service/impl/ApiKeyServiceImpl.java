package com.pei.dehaze.service.impl;

import cn.hutool.core.util.RandomUtil;
import cn.hutool.crypto.SecureUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.mapper.SysApiKeyMapper;
import com.pei.dehaze.model.dto.ApiKeyResult;
import com.pei.dehaze.model.dto.UserAuthInfo;
import com.pei.dehaze.model.entity.SysApiKey;
import com.pei.dehaze.model.form.ApiKeyForm;
import com.pei.dehaze.security.model.SysUserDetails;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.ApiKeyService;
import com.pei.dehaze.service.SysUserService;
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

    @Override
    public ApiKeyResult createApiKey(ApiKeyForm form) {
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
        this.save(apiKey);

        return ApiKeyResult.builder()
                .id(apiKey.getId())
                .name(apiKey.getName())
                .apiKey(rawKey)
                .keyPrefix(apiKey.getKeyPrefix())
                .status(apiKey.getStatus())
                .expiresAt(apiKey.getExpiresAt())
                .lastUsedAt(apiKey.getLastUsedAt())
                .createTime(apiKey.getCreateTime())
                .build();
    }

    @Override
    public List<ApiKeyResult> listApiKeys() {
        Long userId = SecurityUtils.getUserId();
        List<SysApiKey> keys = this.list(new LambdaQueryWrapper<SysApiKey>()
                .eq(SysApiKey::getUserId, userId)
                .orderByDesc(SysApiKey::getCreateTime));
        return keys.stream().map(k -> ApiKeyResult.builder()
                .id(k.getId())
                .name(k.getName())
                .keyPrefix(k.getKeyPrefix())
                .status(k.getStatus())
                .expiresAt(k.getExpiresAt())
                .lastUsedAt(k.getLastUsedAt())
                .createTime(k.getCreateTime())
                .build()).collect(Collectors.toList());
    }

    @Override
    public boolean deleteApiKey(Long id) {
        Long userId = SecurityUtils.getUserId();
        SysApiKey apiKey = this.getById(id);
        if (apiKey == null || !apiKey.getUserId().equals(userId)) {
            return false;
        }
        return this.removeById(id);
    }

    @Override
    public Authentication authenticateByKey(String rawKey) {
        String keyHash = SecureUtil.sha256(rawKey);
        SysApiKey apiKey = this.getOne(new LambdaQueryWrapper<SysApiKey>()
                .eq(SysApiKey::getKeyHash, keyHash));
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
