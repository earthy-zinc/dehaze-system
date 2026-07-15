package com.pei.dehaze.converter;

import com.pei.dehaze.model.entity.SysAlgorithmVersion;
import com.pei.dehaze.model.vo.AlgorithmVersionVO;
import org.mapstruct.Mapper;

/**
 * 算法版本对象转换器
 *
 * @author earthy-zinc
 * @since 2024-06-12
 */
@Mapper(componentModel = "spring")
public interface AlgorithmVersionConverter {

    AlgorithmVersionVO entity2Vo(SysAlgorithmVersion entity);
}
