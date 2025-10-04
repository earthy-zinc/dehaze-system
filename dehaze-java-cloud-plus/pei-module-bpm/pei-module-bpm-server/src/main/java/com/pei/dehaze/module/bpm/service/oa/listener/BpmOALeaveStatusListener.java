package com.pei.dehaze.module.bpm.service.oa.listener;

import com.pei.dehaze.module.bpm.api.event.BpmProcessInstanceStatusEvent;
import com.pei.dehaze.module.bpm.api.event.BpmProcessInstanceStatusEventListener;
import com.pei.dehaze.module.bpm.service.oa.BpmOALeaveService;
import com.pei.dehaze.module.bpm.service.oa.BpmOALeaveServiceImpl;
import jakarta.annotation.Resource;
import org.springframework.stereotype.Component;

/**
 * OA 请假单的结果的监听器实现类
 *
 * @author earthyzinc
 */
@Component
public class BpmOALeaveStatusListener extends BpmProcessInstanceStatusEventListener {

    @Resource
    private BpmOALeaveService leaveService;

    @Override
    protected String getProcessDefinitionKey() {
        return BpmOALeaveServiceImpl.PROCESS_KEY;
    }

    @Override
    protected void onEvent(BpmProcessInstanceStatusEvent event) {
        leaveService.updateLeaveStatus(Long.parseLong(event.getBusinessKey()), event.getStatus());
    }

}
