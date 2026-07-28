package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
@Schema(description = "自动续费配置VO")
public class AutoRenewConfigVO {

    @Schema(description = "用户ID")
    private Long userId;

    @Schema(description = "套餐ID")
    private Long packageId;

    @Schema(description = "套餐名称")
    private String packageName;

    @Schema(description = "支付方式")
    private String payMethod;

    @Schema(description = "是否启用")
    private Boolean enabled;

    @Schema(description = "下次扣款时间")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime nextRenewTime;

    @Schema(description = "连续失败次数")
    private Integer failCount;

    @Schema(description = "关闭原因")
    private String closeReason;
}
