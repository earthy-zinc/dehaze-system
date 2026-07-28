package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDate;

@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
@Schema(description = "签到结果视图对象")
public class SignInResultVO {

    @Schema(description = "签到日期")
    @JsonFormat(pattern = "yyyy-MM-dd")
    private LocalDate signDate;

    @Schema(description = "连续签到天数")
    private Integer continuousDays;

    @Schema(description = "本次获得成长值")
    private Integer growthValue;

    @Schema(description = "连续签到奖励成长值")
    private Integer bonusGrowth;
}
