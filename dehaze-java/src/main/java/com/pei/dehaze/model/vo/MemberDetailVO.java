package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.time.LocalDateTime;

@Data
@EqualsAndHashCode(callSuper = true)
@JsonInclude(JsonInclude.Include.NON_NULL)
@Schema(description = "会员详情视图对象（后台）")
public class MemberDetailVO extends MemberProfileVO {

    @Schema(description = "等级来源(growth:成长值达标;purchase:套餐购买;admin:管理员调整)")
    private String levelSource;

    @Schema(description = "累计消费金额（单位：分）")
    private Long totalConsumption;

    @Schema(description = "首次成为会员时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime becomeMemberTime;

    @Schema(description = "冻结原因")
    private String frozenReason;

    @Schema(description = "冻结时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime frozenTime;

    @Schema(description = "配额所属月份（格式yyyyMM）")
    private Integer quotaResetMonth;
}
