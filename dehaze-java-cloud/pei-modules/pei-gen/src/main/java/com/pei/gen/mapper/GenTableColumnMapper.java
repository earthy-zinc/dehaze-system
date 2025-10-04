package com.pei.gen.mapper;

import com.baomidou.mybatisplus.annotation.InterceptorIgnore;
import com.pei.gen.domain.GenTableColumn;
import com.pei.common.mybatis.core.mapper.BaseMapperPlus;

/**
 * 业务字段 数据层
 *
 * @author Lion Li
 */
@InterceptorIgnore(dataPermission = "true", tenantLine = "true")
public interface GenTableColumnMapper extends BaseMapperPlus<GenTableColumn, GenTableColumn> {

}
