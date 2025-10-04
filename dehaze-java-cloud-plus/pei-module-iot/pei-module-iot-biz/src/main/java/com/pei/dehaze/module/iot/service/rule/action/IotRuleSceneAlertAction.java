package com.pei.dehaze.module.iot.service.rule.action;

import com.pei.dehaze.module.iot.dal.dataobject.rule.IotRuleSceneDO;
import com.pei.dehaze.module.iot.enums.rule.IotRuleSceneActionTypeEnum;
import com.pei.dehaze.module.iot.mq.message.IotDeviceMessage;
import org.springframework.stereotype.Component;

import javax.annotation.Nullable;

/**
 * IoT 告警的 {@link IotRuleSceneAction} 实现类
 *
 * @author earthyzinc
 */
@Component
public class IotRuleSceneAlertAction implements IotRuleSceneAction {

    @Override
    public void execute(@Nullable IotDeviceMessage message, IotRuleSceneDO.ActionConfig config) {
        // TODO @芋艿：待实现
    }

    @Override
    public IotRuleSceneActionTypeEnum getType() {
        return IotRuleSceneActionTypeEnum.ALERT;
    }

}
