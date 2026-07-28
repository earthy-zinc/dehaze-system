package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
@Schema(description = "优惠券模板VO")
public class CouponVO {

    @Schema(description = "优惠券ID")
    private Long id;

    @Schema(description = "优惠券名称")
    private String name;

    @Schema(description = "类型")
    private String type;

    @Schema(description = "面值")
    private Long faceValue;

    @Schema(description = "使用门槛")
    private Long threshold;

    @Schema(description = "有效期类型")
    private String validType;

    @Schema(description = "有效期开始")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime validStart;

    @Schema(description = "有效期结束")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime validEnd;

    @Schema(description = "领取后有效天数")
    private Integer validDays;

    @Schema(description = "发放总量")
    private Integer totalQty;

    @Schema(description = "已发放数量")
    private Integer issuedQty;

    @Schema(description = "已使用数量")
    private Integer usedQty;

    @Schema(description = "每人限领")
    private Integer perUserLimit;

    @Schema(description = "适用套餐ID列表")
    private List<Long> applicableScope;

    @Schema(description = "状态(1:启用;0:禁用)")
    private Integer status;

    @Schema(description = "创建时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
