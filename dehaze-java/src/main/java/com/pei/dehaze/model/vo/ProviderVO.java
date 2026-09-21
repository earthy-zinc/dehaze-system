package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.pei.dehaze.model.form.UserIdentityForwardForm;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.Map;

@Schema(description = "AI 供应商视图")
@Data
public class ProviderVO {

    private Long id;

    private String providerCode;

    private String displayName;

    private String apiBaseUrl;

    private String protocolType;

    private String authType;

    private Map<String, Object> defaultHeaders;

    private Integer sortOrder;

    private Integer healthCheckEnabled;

    private UserIdentityForwardForm userIdentityForward;

    private String remark;

    @Schema(description = "健康状态(healthy:健康;suspicious:可疑;open:熔断)")
    private String health;

    private Integer status;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updateTime;
}
