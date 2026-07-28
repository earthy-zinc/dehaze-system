package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
@Schema(description = "套餐分页VO")
public class PackagePageVO {

    @Schema(description = "套餐ID")
    private Long id;

    @Schema(description = "套餐名称")
    private String name;

    @Schema(description = "会员等级编码")
    private String levelCode;

    @Schema(description = "会员等级名称")
    private String levelName;

    @Schema(description = "计费周期")
    private String period;

    @Schema(description = "有效期天数")
    private Integer periodDays;

    @Schema(description = "原价（分）")
    private Long originalPrice;

    @Schema(description = "促销价（分）")
    private Long salePrice;

    @Schema(description = "日均价格（分）")
    private Long dailyPrice;

    @Schema(description = "销量")
    private Long salesCount;

    @Schema(description = "状态(1:上架;0:下架)")
    private Integer status;

    @Schema(description = "创建时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;
}
