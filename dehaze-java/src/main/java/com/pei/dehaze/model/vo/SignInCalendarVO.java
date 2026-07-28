package com.pei.dehaze.model.vo;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonInclude;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.time.LocalDate;
import java.util.List;

@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
@Schema(description = "签到日历视图对象")
public class SignInCalendarVO {

    @Schema(description = "签到日期列表")
    @JsonFormat(pattern = "yyyy-MM-dd")
    private List<LocalDate> signDates;

    @Schema(description = "连续签到天数")
    private Integer continuousDays;

    @Schema(description = "当月累计签到天数")
    private Integer totalDays;
}
