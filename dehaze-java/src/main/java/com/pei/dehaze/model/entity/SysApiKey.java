package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.io.Serial;
import java.time.LocalDateTime;

/**
 * API 密钥实体
 * <p>
 * 吊销机制：使用 {@link #revokedAt}（NULL=未吊销，非NULL=已吊销）。
 * 本表不使用逻辑删除（无 deleted 字段）：API Key 唯一的"移除"即吊销，
 * 吊销后 hash 必须永久保留以拒绝已泄露的旧密钥，故用 revoked_at 标记而非删除。
 * </p>
 */
@Data
@EqualsAndHashCode(callSuper = false)
public class SysApiKey extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long userId;

    private String name;

    private String keyPrefix;

    private String keyHash;

    private Integer status;

    private LocalDateTime expiresAt;

    private LocalDateTime lastUsedAt;

    /**
     * 吊销时间（NULL 表示未吊销；非 NULL 表示已吊销）
     */
    private LocalDateTime revokedAt;

    @Serial
    @TableField(exist = false)
    private static final long serialVersionUID = 1L;
}
