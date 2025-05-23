package com.pei.system.domain.convert;

import io.github.linpeilie.BaseMapper;
import com.pei.system.api.domain.vo.RemoteClientVo;
import com.pei.system.domain.vo.SysClientVo;
import org.mapstruct.Mapper;
import org.mapstruct.MappingConstants;
import org.mapstruct.ReportingPolicy;

/**
 * 客户端数据转换器
 *
 * @author Michelle.Chung
 */
@Mapper(componentModel = MappingConstants.ComponentModel.SPRING, unmappedTargetPolicy = ReportingPolicy.IGNORE)
public interface SysClientVoConvert extends BaseMapper<SysClientVo, RemoteClientVo> {
}
