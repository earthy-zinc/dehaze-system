package com.pei.dehaze.module.iot.controller.admin.rule.vo.databridge;

import com.pei.dehaze.framework.common.enums.CommonStatusEnum;
import com.pei.dehaze.framework.common.pojo.PageParam;
import com.pei.dehaze.framework.common.validation.InEnum;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;

import java.time.LocalDateTime;

import static com.pei.dehaze.framework.common.util.date.DateUtils.FORMAT_YEAR_MONTH_DAY_HOUR_MINUTE_SECOND;

@Schema(description = "管理后台 - IoT 数据桥梁分页 Request VO")
@Data
public class IotDataBridgePageReqVO extends PageParam {

    @Schema(description = "桥梁名称", example = "赵六")
    private String name;

    @Schema(description = "桥梁状态", example = "1")
    @InEnum(CommonStatusEnum.class)
    private Integer status;

    @Schema(description = "创建时间")
    @DateTimeFormat(pattern = FORMAT_YEAR_MONTH_DAY_HOUR_MINUTE_SECOND)
    private LocalDateTime[] createTime;

}
