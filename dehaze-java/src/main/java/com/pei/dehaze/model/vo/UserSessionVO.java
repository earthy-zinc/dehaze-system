package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Builder;
import lombok.Data;

/**
 * 在线会话视图对象
 */
@Schema(description = "在线会话")
@Data
@Builder
public class UserSessionVO {

    @Schema(description = "会话ID")
    private String sessionId;

    @Schema(description = "用户名")
    private String username;

    @Schema(description = "设备类型(web/android/flutter/miniprogram)")
    private String deviceType;

    @Schema(description = "登录时间")
    private String loginTime;

    @Schema(description = "登录IP")
    private String ip;

    @Schema(description = "最后访问时间")
    private String lastAccessTime;
}
