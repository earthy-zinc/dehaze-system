package com.pei.dehaze.module.trade.job.order;

import com.pei.dehaze.framework.tenant.core.job.TenantJob;
import com.pei.dehaze.module.trade.service.order.TradeOrderUpdateService;
import com.xxl.job.core.handler.annotation.XxlJob;
import jakarta.annotation.Resource;
import org.springframework.stereotype.Component;

/**
 * 交易订单的自动过期 Job
 *
 * @author earthyzinc
 */
@Component
public class TradeOrderAutoCancelJob {

    @Resource
    private TradeOrderUpdateService tradeOrderUpdateService;

    @XxlJob("tradeOrderAutoCancelJob")
    @TenantJob // 多租户
    public String execute() {
        int count = tradeOrderUpdateService.cancelOrderBySystem();
        return String.format("过期订单 %s 个", count);
    }

}
