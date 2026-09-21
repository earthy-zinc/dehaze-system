package com.pei.dehaze.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.baomidou.mybatisplus.extension.handlers.JacksonTypeHandler;
import com.pei.dehaze.common.base.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.io.Serial;
import java.time.LocalDateTime;
import java.util.List;

/**
 * API 密钥实体
 * <p>
 * 吊销机制：使用 {@link #revokedAt}（NULL=未吊销，非NULL=已吊销）。
 * 本表不使用逻辑删除（无 deleted 字段）：API Key 唯一的"移除"即吊销，
 * 吊销后 hash 必须永久保留以拒绝已泄露的旧密钥，故用 revoked_at 标记而非删除。
 * </p>
 * <p>
 * autoResultMap 必须开启：modelWhitelist 为 JSON 列，只有走 resultMap 才会经
 * {@link JacksonTypeHandler} 反序列化（否则 select 取到的是未转换的 JSON 串）。
 * </p>
 */
@Data
@EqualsAndHashCode(callSuper = false)
@TableName(value = "sys_api_key", autoResultMap = true)
public class SysApiKey extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;

    private Long userId;

    private String name;

    private String keyPrefix;

    private String keyHash;

    private Integer status;

    private LocalDateTime expiresAt;

    private Long dailyQuota;

    private Long monthlyQuota;

    private Long rpmLimit;

    /**
     * 模型白名单（NULL 或空数组 = 继承用户可见模型）。
     * 执行点只在 python 兼容层 compatible_governance，go/java 转发即受治，不建本地执行点（防双重计数）。
     */
    @TableField(typeHandler = JacksonTypeHandler.class)
    private List<String> modelWhitelist;

    private LocalDateTime lastUsedAt;

    /**
     * 吊销时间（NULL 表示未吊销；非 NULL 表示已吊销）
     */
    private LocalDateTime revokedAt;

    @Serial
    @TableField(exist = false)
    private static final long serialVersionUID = 1L;
}
