package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
@Schema(description = "用户优惠券VO")
public class UserCouponVO {

    @Schema(description = "实例ID")
    private Long id;

    @Schema(description = "优惠券模板ID")
    private Long couponId;

    @Schema(description = "优惠券名称")
    private String couponName;

    @Schema(description = "类型")
    private String type;

    @Schema(description = "面值")
    private Long faceValue;

    @Schema(description = "使用门槛")
    private Long threshold;

    @Schema(description = "状态(1:未使用;2:已使用;3:已过期;4:已锁定)")
    private Integer status;

    @Schema(description = "领取时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime receiveTime;

    @Schema(description = "过期时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime expireTime;

    @Schema(description = "使用时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime usedTime;

    @Schema(description = "使用的订单ID")
    private Long usedOrderId;

    @Schema(description = "适用套餐ID列表")
    private List<Long> applicableScope;
}
