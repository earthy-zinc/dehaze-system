package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

/**
 * 启用供应商列表（/providers/enabled，登录用户可见）的精简视图：
 * 不含 api_base_url/default_headers/user_identity_forward/remark 等供应商内部配置。
 */
@Schema(description = "启用供应商精简视图")
@Data
public class ProviderEnabledVO {

    private Long id;

    private String providerCode;

    private String displayName;

    private String protocolType;

    private String health;

    private Integer status;
}
