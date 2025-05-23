package com.pei.workflow.dubbo;

import lombok.RequiredArgsConstructor;
import org.apache.dubbo.config.annotation.DubboService;
import com.pei.workflow.api.RemoteWorkflowService;
import com.pei.workflow.api.domain.RemoteCompleteTask;
import com.pei.workflow.api.domain.RemoteStartProcess;
import com.pei.workflow.api.domain.RemoteStartProcessReturn;
import com.pei.workflow.service.WorkflowService;

import java.util.List;
import java.util.Map;

/**
 * RemoteWorkflowServiceImpl
 *
 * @Author ZETA
 * @Date 2024/6/3
 */
@DubboService
@RequiredArgsConstructor
public class RemoteWorkflowServiceImpl implements RemoteWorkflowService {

    private final WorkflowService workflowService;

    @Override
    public boolean deleteInstance(List<Long> businessIds) {
        return workflowService.deleteInstance(businessIds);
    }

    @Override
    public String getBusinessStatusByTaskId(Long taskId) {
        return workflowService.getBusinessStatusByTaskId(taskId);
    }

    @Override
    public String getBusinessStatus(String businessId) {
        return workflowService.getBusinessStatus(businessId);
    }

    @Override
    public void setVariable(Long instanceId, Map<String, Object> variable) {
        workflowService.setVariable(instanceId, variable);
    }

    @Override
    public Map<String, Object> instanceVariable(Long instanceId) {
        return workflowService.instanceVariable(instanceId);
    }

    @Override
    public Long getInstanceIdByBusinessId(String businessId) {
        return workflowService.getInstanceIdByBusinessId(businessId);
    }

    @Override
    public void syncDef(String tenantId) {
        workflowService.syncDef(tenantId);
    }

    @Override
    public RemoteStartProcessReturn startWorkFlow(RemoteStartProcess startProcess) {
        return workflowService.startWorkFlow(startProcess);
    }

    @Override
    public boolean completeTask(RemoteCompleteTask completeTask) {
        return workflowService.completeTask(completeTask);
    }

}
