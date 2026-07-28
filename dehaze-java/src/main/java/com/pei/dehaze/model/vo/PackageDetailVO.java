package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.List;
import java.util.Map;

@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
@Schema(description = "套餐详情VO")
public class PackageDetailVO {

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

    @Schema(description = "套餐描述")
    private String description;

    @Schema(description = "实际生效权益")
    private Map<String, Integer> benefits;

    @Schema(description = "进行中的促销活动")
    private List<PromotionVO> activePromotions;

    @Schema(description = "销量")
    private Long salesCount;
}
