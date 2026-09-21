package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Data
@Schema(description = "余额退款审核表单")
public class BalanceRefundAuditForm {

    @Schema(description = "审核备注")
    private String remark;

    @Schema(description = "原路退回渠道(wechat/alipay)，无法原路由管理员指定")
    private String channel;
}
