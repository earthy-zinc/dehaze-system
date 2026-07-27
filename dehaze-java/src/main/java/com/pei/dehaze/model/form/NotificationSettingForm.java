package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.Map;

@Data
@Schema(description = "通知偏好设置表单")
public class NotificationSettingForm {

    @Schema(description = "APP推送总开关")
    private Boolean pushEnabled;

    @Schema(description = "免打扰开关")
    private Boolean dndEnabled;

    @Schema(description = "免打扰开始时间(HH:mm:ss)")
    private String dndStart;

    @Schema(description = "免打扰结束时间(HH:mm:ss)")
    private String dndEnd;

    @Schema(description = "细粒度偏好")
    private Map<String, Object> preferences;
}
