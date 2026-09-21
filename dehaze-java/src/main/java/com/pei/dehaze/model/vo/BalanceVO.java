package com.pei.dehaze.model.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Data
@Schema(description = "余额账户VO")
public class BalanceVO {

    @Schema(description = "可用余额(分)")
    private Long balance;

    @Schema(description = "冻结余额(分)")
    private Long frozenBalance;
}
