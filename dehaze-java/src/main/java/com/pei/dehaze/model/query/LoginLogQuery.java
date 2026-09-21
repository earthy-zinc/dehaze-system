package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * 登录日志分页查询对象
 */
@Schema(description = "登录日志分页查询对象")
@Data
@EqualsAndHashCode(callSuper = true)
public class LoginLogQuery extends BasePageQuery {

    @Schema(description = "按用户名筛选（精确匹配）")
    private String username;

    @Schema(description = "按IP筛选（精确匹配）")
    private String ip;

    @Schema(description = "登录状态(1:成功;0:失败)")
    private Integer status;

    @Schema(description = "设备类型(web/android/flutter/miniprogram)")
    private String deviceType;

    @Schema(description = "开始时间")
    private String startTime;

    @Schema(description = "结束时间")
    private String endTime;
}
