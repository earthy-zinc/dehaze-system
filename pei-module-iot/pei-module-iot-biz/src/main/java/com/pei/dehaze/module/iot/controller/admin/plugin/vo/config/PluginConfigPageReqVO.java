package com.pei.dehaze.module.iot.controller.admin.plugin.vo.config;

import com.pei.dehaze.framework.common.pojo.PageParam;
import com.pei.dehaze.framework.common.validation.InEnum;
import com.pei.dehaze.module.iot.enums.plugin.IotPluginStatusEnum;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Schema(description = "管理后台 - IoT 插件配置分页 Request VO")
@Data
public class PluginConfigPageReqVO extends PageParam {

    @Schema(description = "插件名称", example = "http")
    private String name;

    @Schema(description = "状态", example = "1")
    @InEnum(IotPluginStatusEnum.class)
    private Integer status;

}
