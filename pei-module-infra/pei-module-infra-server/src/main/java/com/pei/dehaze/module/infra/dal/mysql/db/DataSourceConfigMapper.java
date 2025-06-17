package com.pei.dehaze.module.infra.dal.mysql.db;

import com.pei.dehaze.framework.mybatis.core.mapper.BaseMapperX;
import com.pei.dehaze.module.infra.dal.dataobject.db.DataSourceConfigDO;
import org.apache.ibatis.annotations.Mapper;

/**
 * 数据源配置 Mapper
 *
 * @author earthyzinc
 */
@Mapper
public interface DataSourceConfigMapper extends BaseMapperX<DataSourceConfigDO> {
}
