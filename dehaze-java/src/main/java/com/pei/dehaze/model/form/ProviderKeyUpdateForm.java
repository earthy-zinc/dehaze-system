package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.Size;
import lombok.Data;

import java.time.LocalDateTime;

/** 供应商 API Key 更新表单（null 视为不修改；Key 明文不可改） */
@Schema(description = "供应商 API Key 更新表单")
@Data
public class ProviderKeyUpdateForm {

    @Size(min = 1, max = 128)
    private String name;

    private Integer priority;

    @Min(1)
    private Integer weight;

    private Integer status;

    @Min(1)
    private Integer dailyQuota;

    @Min(0)
    private Integer rpmLimit;

    private LocalDateTime expiresAt;
}
