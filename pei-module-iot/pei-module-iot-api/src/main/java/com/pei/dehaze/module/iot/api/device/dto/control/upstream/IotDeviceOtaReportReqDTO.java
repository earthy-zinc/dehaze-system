package com.pei.dehaze.module.iot.api.device.dto.control.upstream;

// TODO @芋艿：待实现：/ota/${productKey}/${deviceName}/report
/**
 * IoT 设备【OTA】上报 Request DTO（上报固件版本）
 *
 * @author earthyzinc
 */
public class IotDeviceOtaReportReqDTO {

    /**
     * 固件编号
     */
    private Long firmwareId;

    /**
     * 固件版本
     */
    private String version;

}
