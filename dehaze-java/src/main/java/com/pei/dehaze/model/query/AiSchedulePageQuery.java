package com.pei.dehaze.model.query;

import com.pei.dehaze.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * 定时任务分页查询参数
 *
 * @author dehaze
 */
@Data
@EqualsAndHashCode(callSuper = true)
@Schema(description = "定时任务分页查询参数")
public class AiSchedulePageQuery extends BasePageQuery {

    @Schema(description = "关键字(按名称模糊搜索)")
    private String keyword;
}
