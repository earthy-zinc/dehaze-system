package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import java.math.BigDecimal;
import lombok.Data;

@Schema(description = "管理员手动调整积分表单")
@Data
public class AiBillingAdjustForm {

    @NotNull(message = "用户不能为空")
    private Long userId;

    /** 调整积分（正数增加;负数扣减），不可为 0；整数积分，小数由服务层拒绝 */
    @NotNull(message = "调整积分数不能为空")
    private BigDecimal amount;

    @NotBlank(message = "调整原因不能为空")
    private String reason;
}
