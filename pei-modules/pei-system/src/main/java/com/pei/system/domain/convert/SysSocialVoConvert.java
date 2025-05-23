package com.pei.system.domain.convert;

import com.pei.system.domain.vo.SysSocialVo;
import io.github.linpeilie.BaseMapper;
import com.pei.system.api.domain.vo.RemoteSocialVo;
import org.mapstruct.Mapper;
import org.mapstruct.MappingConstants;
import org.mapstruct.ReportingPolicy;

/**
 * 社交数据转换器
 *
 * @author Michelle.Chung
 */
@Mapper(componentModel = MappingConstants.ComponentModel.SPRING, unmappedTargetPolicy = ReportingPolicy.IGNORE)
public interface SysSocialVoConvert extends BaseMapper<SysSocialVo, RemoteSocialVo> {
}
