package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Schema(description = "成本单价版本创建表单")
@Data
public class AiModelCostForm {

    @NotBlank(message = "模型标识不能为空")
    private String modelId;

    @NotNull(message = "供应商不能为空")
    private Long providerId;

    private String currency = "CNY";

    private LocalDateTime effectiveFrom;

    private LocalDateTime effectiveTo;

    private Integer status = 1;

    private List<Detail> details = new ArrayList<>();

    @Schema(description = "成本档位（token 类型 × 时段 × 上下文分段）")
    @Data
    public static class Detail {

        private String tokenType;

        private String timeSlot;

        private Long minTokens = 0L;

        private Long maxTokens;

        private BigDecimal unitPrice;
    }
}
