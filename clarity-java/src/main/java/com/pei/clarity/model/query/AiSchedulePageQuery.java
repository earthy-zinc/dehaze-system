package com.pei.clarity.model.query;

import com.pei.clarity.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * 定时任务分页查询参数
 *
 * @author clarity
 */
@Data
@EqualsAndHashCode(callSuper = true)
@Schema(description = "定时任务分页查询参数")
public class AiSchedulePageQuery extends BasePageQuery {

    @Schema(description = "关键字(按名称模糊搜索)")
    private String keyword;
}
