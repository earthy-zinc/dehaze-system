package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.Map;

@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
@Schema(description = "促销活动VO")
public class PromotionVO {

    @Schema(description = "活动ID")
    private Long id;

    @Schema(description = "活动名称")
    private String name;

    @Schema(description = "活动类型")
    private String type;

    @Schema(description = "活动描述")
    private String description;

    @Schema(description = "开始时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime startTime;

    @Schema(description = "结束时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime endTime;

    @Schema(description = "活动规则")
    private Map<String, Object> activityRules;

    @Schema(description = "是否新用户专享(0:否;1:是)")
    private Integer newUserOnly;

    @Schema(description = "状态(1:启用;0:禁用)")
    private Integer status;
}
