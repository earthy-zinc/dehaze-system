package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
@Schema(description = "成长值流水视图对象")
public class GrowthLogVO {

    @Schema(description = "流水ID")
    private Long id;

    @Schema(description = "变动类型(dehaze/evaluate/rating/sign_in/sign_in_bonus/consume/refund_deduct/admin_adjust)")
    private String changeType;

    @Schema(description = "变动类型描述")
    private String changeTypeLabel;

    @Schema(description = "变动值（正数增加/负数扣减）")
    private Integer changeValue;

    @Schema(description = "变动后成长值余额")
    private Long balance;

    @Schema(description = "关联业务ID（订单号/任务ID/签到记录ID）")
    private String relatedId;

    @Schema(description = "变动原因")
    private String reason;

    @Schema(description = "操作人ID（仅管理员调整时记录）")
    private Long operatorId;

    @Schema(description = "创建时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
