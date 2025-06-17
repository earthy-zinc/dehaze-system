package com.pei.dehaze.module.iot.mq.consumer.device;

import com.pei.dehaze.module.iot.mq.message.IotDeviceMessage;
import com.pei.dehaze.module.iot.service.device.data.IotDeviceLogService;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.event.EventListener;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;

/**
 * 针对 {@link IotDeviceMessage} 的消费者，记录设备日志
 *
 * @author earthyzinc
 */
@Component
@Slf4j
public class IotDeviceLogMessageConsumer {

    @Resource
    private IotDeviceLogService deviceLogService;

    @EventListener
    @Async
    public void onMessage(IotDeviceMessage message) {
        log.info("[onMessage][消息内容({})]", message);
        deviceLogService.createDeviceLog(message);
    }

}
