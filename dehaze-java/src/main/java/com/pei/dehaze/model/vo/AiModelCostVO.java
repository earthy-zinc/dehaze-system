package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import com.fasterxml.jackson.databind.ser.std.ToStringSerializer;
import io.swagger.v3.oas.annotations.media.Schema;
import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import lombok.Data;

/** AI 模型成本单价版本（含档位明细） */
@Schema(description = "AI 模型成本单价版本")
@Data
public class AiModelCostVO {

    private Long id;

    private String modelId;

    private Long providerId;

    private Integer priceVersion;

    private String currency;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime effectiveFrom;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime effectiveTo;

    private Integer status;

    private List<Detail> details = new ArrayList<>();

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime createTime;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime updateTime;

    @Schema(description = "成本档位明细")
    @Data
    public static class Detail {

        private Long id;

        private Long priceId;

        private String tokenType;

        private String timeSlot;

        private Long minTokens;

        private Long maxTokens;

        /** 单价：python `ModelCostDetailResult.unit_price` 为 Decimal → JSON 字符串，前端计算前需转 Number */
        @JsonSerialize(using = ToStringSerializer.class)
        private BigDecimal unitPrice;
    }
}
