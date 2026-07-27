package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.Map;

@Data
@Schema(description = "通知偏好设置视图对象")
public class NotificationSettingsVO {

    @Schema(description = "APP推送总开关")
    private Boolean pushEnabled;

    @Schema(description = "免打扰开关")
    private Boolean dndEnabled;

    @Schema(description = "免打扰开始时间")
    private String dndStart;

    @Schema(description = "免打扰结束时间")
    private String dndEnd;

    @Schema(description = "细粒度偏好")
    private Map<String, Object> preferences;
}
