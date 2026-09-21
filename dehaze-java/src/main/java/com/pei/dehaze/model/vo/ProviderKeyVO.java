package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

/** 供应商 API Key 视图：不含明文与密文，仅前缀展示 */
@Schema(description = "供应商 API Key 视图")
@Data
public class ProviderKeyVO {

    private Long id;

    private Long providerId;

    private String name;

    private String keyPrefix;

    private Integer status;

    private Integer priority;

    private Integer weight;

    private Integer dailyQuota;

    private Integer rpmLimit;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime expiresAt;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime lastUsedAt;

    private Long lastUsedBy;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updateTime;
}
