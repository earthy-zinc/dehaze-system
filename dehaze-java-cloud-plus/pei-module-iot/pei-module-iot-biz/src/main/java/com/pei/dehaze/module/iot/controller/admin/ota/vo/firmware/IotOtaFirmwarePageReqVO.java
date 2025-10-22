package com.pei.dehaze.module.iot.controller.admin.ota.vo.firmware;

import com.pei.dehaze.framework.common.pojo.PageParam;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Data
@Schema(description = "管理后台 - IoT OTA 固件分页 Request VO")
public class IotOtaFirmwarePageReqVO extends PageParam {

    /**
     * 固件名称
     */
    @Schema(description = "固件名称", example = "智能开关固件")
    private String name;

    /**
     * 产品标识
     */
    @Schema(description = "产品标识", example = "1024")
    private Long productId;

}
