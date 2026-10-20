package com.pei.clarity.model.query;

import com.pei.clarity.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Schema(description = "字典类型分页查询对象")
@Data
@EqualsAndHashCode(callSuper = true)
public class DictTypePageQuery extends BasePageQuery {

    @Schema(description = "关键字(类型名称/类型编码)")
    private String keywords;
}
