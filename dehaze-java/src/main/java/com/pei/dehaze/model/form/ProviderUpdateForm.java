package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Size;
import lombok.Data;

import java.util.Map;

/** AI 供应商更新表单（null 视为不修改；provider_code 为白名单业务键，不可改） */
@Schema(description = "AI 供应商更新表单")
@Data
public class ProviderUpdateForm {

    @Size(min = 1, max = 128)
    private String displayName;

    @Size(min = 1, max = 512)
    private String apiBaseUrl;

    @Size(max = 32)
    private String protocolType;

    @Size(max = 32)
    private String authType;

    private Map<String, Object> defaultHeaders;

    private Integer sortOrder;

    private Integer healthCheckEnabled;

    private UserIdentityForwardForm userIdentityForward;

    private String remark;

    private Integer status;
}
