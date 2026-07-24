package com.pei.dehaze.sdk.model.apikey;

import lombok.Data;

@Data
public class ApiKeyInfo {
    private Long id;
    private String name;
    private String apiKey;
    private String keyPrefix;
    private Integer status;
    private String expiresAt;
    private String lastUsedAt;
    private String createTime;
}
