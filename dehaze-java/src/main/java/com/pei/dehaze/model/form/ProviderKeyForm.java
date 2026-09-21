package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;

import java.time.LocalDateTime;

@Schema(description = "供应商 API Key 创建表单")
@Data
public class ProviderKeyForm {

    @NotBlank(message = "Key 名称不能为空")
    @Size(min = 1, max = 128)
    private String name;

    @NotBlank(message = "Key 明文不能为空")
    private String key;

    private Integer priority = 0;

    @Min(1)
    private Integer weight = 1;

    @Min(1)
    private Integer dailyQuota;

    @Min(0)
    private Integer rpmLimit;

    private LocalDateTime expiresAt;

    private Integer status = 1;
}
