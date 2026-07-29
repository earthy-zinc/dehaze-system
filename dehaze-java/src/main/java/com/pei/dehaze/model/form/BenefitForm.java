package com.pei.dehaze.model.form;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import lombok.Data;

@Data
@Schema(description = "会员权益配置表单")
public class BenefitForm {

    @Schema(description = "等级名称")
    private String levelName;

    @Schema(description = "成长值下限")
    @Min(value = 0, message = "成长值下限不能为负数")
    private Long growthMin;

    @Schema(description = "成长值上限（0表示无上限）")
    @Min(value = 0, message = "成长值上限不能为负数")
    private Long growthMax;

    @Schema(description = "月度去雾次数配额")
    @Min(value = 0, message = "月度去雾次数配额不能为负数")
    private Integer monthlyDehazeQuota;

    @Schema(description = "月度评估次数配额")
    @Min(value = 0, message = "月度评估次数配额不能为负数")
    private Integer monthlyEvaluateQuota;

    @Schema(description = "历史记录保留条数")
    @Min(value = 0, message = "历史记录保留条数不能为负数")
    private Integer historyRetention;

    @Schema(description = "批量处理上限（张）")
    @Min(value = 0, message = "批量处理上限不能为负数")
    private Integer batchLimit;

    @Schema(description = "处理优先级(1:普通;2:优先;3:高优先;4:最高)")
    @Min(value = 1, message = "处理优先级最小为1")
    @Max(value = 4, message = "处理优先级最大为4")
    private Integer priority;

    @Schema(description = "高级参数调节(0:关闭;1:开启)")
    @Min(value = 0, message = "高级参数调节值非法")
    @Max(value = 1, message = "高级参数调节值非法")
    private Integer advancedParams;

    @Schema(description = "高清图导出(0:关闭;1:开启)")
    @Min(value = 0, message = "高清图导出值非法")
    @Max(value = 1, message = "高清图导出值非法")
    private Integer hdExport;

    @Schema(description = "对比报告导出(0:关闭;1:开启)")
    @Min(value = 0, message = "对比报告导出值非法")
    @Max(value = 1, message = "对比报告导出值非法")
    private Integer reportExport;

    @Schema(description = "批量打包下载(0:关闭;1:开启)")
    @Min(value = 0, message = "批量打包下载值非法")
    @Max(value = 1, message = "批量打包下载值非法")
    private Integer batchDownload;

    @Schema(description = "排序值")
    @Min(value = 0, message = "排序值不能为负数")
    private Integer sort;

    @Schema(description = "状态(1:启用;0:禁用)")
    @Min(value = 0, message = "状态值非法")
    @Max(value = 1, message = "状态值非法")
    private Integer status;
}
