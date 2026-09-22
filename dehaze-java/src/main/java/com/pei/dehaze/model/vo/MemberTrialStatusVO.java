package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Data
@Schema(description = "试用开通状态视图对象")
public class MemberTrialStatusVO {

    @Schema(description = "是否展示试用入口")
    private Boolean showTrialEntry;

    @Schema(description = "试用天数")
    private Integer trialDays;

    @Schema(description = "试用赠送积分")
    private Integer trialCredits;

    @Schema(description = "体验券是否已激活")
    private Boolean voucherActivated;

    /**
     * 未激活体验券时显式返回 null：全局 serialization-inclusion=non_null 会连键一起剔除，
     * 前端无法区分"无到期时间"与"字段缺失"
     */
    @Schema(description = "体验券到期时间")
    @JsonInclude(JsonInclude.Include.ALWAYS)
    private String voucherExpireTime;

    @Schema(description = "AI 试用积分余额")
    private Long aiTrialCreditsBalance;

    @Schema(description = "新用户专享是否可用")
    private Boolean newUserExclusiveAvailable;

    @Schema(description = "是否付费会员")
    private Boolean paidMembership;
}
