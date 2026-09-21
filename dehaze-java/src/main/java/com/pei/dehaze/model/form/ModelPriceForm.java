package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Schema(description = "模型用户售价版本创建表单")
@Data
public class ModelPriceForm {

    @NotBlank(message = "模型标识不能为空")
    @Size(min = 1, max = 64)
    private String modelId;

    @NotNull(message = "供应商不能为空")
    private Long providerId;

    @Size(max = 24)
    private String unit = "credits_per_million";

    private LocalDateTime effectiveFrom;

    private LocalDateTime effectiveTo;

    private Integer status = 1;

    private List<Detail> details = new ArrayList<>();

    @Schema(description = "价格档位（token 类型 × 时段 × 上下文分段）")
    @Data
    public static class Detail {

        private String tokenType;

        private String timeSlot;

        @Min(0)
        private Long minTokens = 0L;

        private Long maxTokens;

        @DecimalMin("0")
        private BigDecimal unitPrice;
    }
}
