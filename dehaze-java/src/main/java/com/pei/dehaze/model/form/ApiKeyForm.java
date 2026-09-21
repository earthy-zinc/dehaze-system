package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

@Data
public class ApiKeyForm {

    @NotBlank
    private String name;

    private LocalDateTime expiresAt;

    @Schema(description = "日调用配额(不限制则不传或传0)")
    @Min(value = 1, message = "日调用配额必须大于0")
    private Long dailyQuota;

    @Schema(description = "月调用配额(不限制则不传或传0)")
    @Min(value = 1, message = "月调用配额必须大于0")
    private Long monthlyQuota;

    @Schema(description = "每分钟请求数上限RPM(不限制则不传或传0)")
    @Min(value = 1, message = "RPM上限必须大于0")
    private Long rpmLimit;

    @Schema(description = "模型白名单(不限制或继承用户可见模型则不传或传空数组)")
    private List<String> modelWhitelist;
}
