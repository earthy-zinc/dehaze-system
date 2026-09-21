package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

/**
 * 登录日志视图对象
 */
@Schema(description = "登录日志")
@Data
public class LoginLogVO {

    @Schema(description = "日志ID")
    private String id;

    @Schema(description = "用户ID")
    private Long userId;

    @Schema(description = "用户名")
    private String username;

    @Schema(description = "登录IP")
    private String ip;

    @Schema(description = "登录地点")
    private String location;

    @Schema(description = "浏览器")
    private String browser;

    @Schema(description = "操作系统")
    private String os;

    @Schema(description = "设备类型(web/android/flutter/miniprogram)")
    private String deviceType;

    @Schema(description = "登录状态(1:成功;0:失败)")
    private Integer status;

    @Schema(description = "提示信息")
    private String message;

    @Schema(description = "登录时间")
    private String loginTime;
}
