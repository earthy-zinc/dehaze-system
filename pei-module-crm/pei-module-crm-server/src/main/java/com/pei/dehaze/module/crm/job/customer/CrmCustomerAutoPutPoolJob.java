package com.pei.dehaze.module.crm.job.customer;

import com.pei.dehaze.framework.tenant.core.job.TenantJob;
import com.pei.dehaze.module.crm.service.customer.CrmCustomerService;
import com.xxl.job.core.handler.annotation.XxlJob;
import jakarta.annotation.Resource;
import org.springframework.stereotype.Component;

/**
 * 客户自动掉入公海 Job
 *
 * @author earthyzinc
 */
@Component
public class CrmCustomerAutoPutPoolJob {

    @Resource
    private CrmCustomerService customerService;

    @XxlJob("customerAutoPutPoolJob")
    @TenantJob
    public String execute() {
        int count = customerService.autoPutCustomerPool();
        return String.format("掉入公海客户 %s 个", count);
    }

}
