package com.pei.dehaze.sdk.model.apikey;

import lombok.Data;

@Data
public class ApiKeyCreateRequest {
    private String name;
    private String expiresAt;
}
