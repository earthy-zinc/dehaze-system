package com.pei.dehaze.model.dto;

import lombok.Builder;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@Builder
public class ApiKeyResult {

    private Long id;

    private String name;

    private String apiKey;

    private String keyPrefix;

    private Integer status;

    private LocalDateTime expiresAt;

    private LocalDateTime lastUsedAt;

    private LocalDateTime createTime;
}
