package com.pei.dehaze.model.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import java.time.LocalDateTime;
import java.util.List;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ApiKeyResult {

    private Long id;

    private String name;

    private String apiKey;

    private String keyPrefix;

    private Integer status;

    private Long dailyQuota;

    private Long monthlyQuota;

    private Long rpmLimit;

    private List<String> modelWhitelist;

    private LocalDateTime expiresAt;

    private LocalDateTime lastUsedAt;

    private LocalDateTime createTime;
}
