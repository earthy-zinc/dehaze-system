package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.time.LocalDateTime;

/**
 * 供应商 API Key（表 sys_ai_provider_key）。
 *
 * <p>Key 管理为状态控制（启用/禁用/过期），不使用逻辑删除，删除即物理删除。
 * key_cipher 为 AES-256-CBC 密文（base64），与 Python 端同密钥同算法，跨端可互相解密。
 */
@Data
@EqualsAndHashCode(callSuper = false)
public class SysAiProviderKey extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long providerId;

    private String name;

    private String keyHash;

    private String keyPrefix;

    private String keyCipher;

    private Integer status;

    private Integer priority;

    private Integer weight;

    private Integer dailyQuota;

    private Integer rpmLimit;

    private LocalDateTime expiresAt;

    private LocalDateTime lastUsedAt;

    private Long lastUsedBy;
}
