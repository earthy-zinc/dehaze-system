package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import com.fasterxml.jackson.databind.ser.std.ToStringSerializer;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.List;

@Schema(description = "模型用户售价版本视图")
@Data
public class ModelPriceVO {

    private Long id;

    private String modelId;

    private Long providerId;

    private Integer priceVersion;

    private String unit;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime effectiveFrom;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime effectiveTo;

    private Integer status;

    private List<Detail> details;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updateTime;

    @Schema(description = "价格档位明细")
    @Data
    public static class Detail {

        private Long id;

        private Long priceId;

        private String tokenType;

        private String timeSlot;

        private Long minTokens;

        private Long maxTokens;

        /** 后端 Decimal 序列化为字符串，前端计算前需转 Number */
        @JsonSerialize(using = ToStringSerializer.class)
        private BigDecimal unitPrice;
    }
}
