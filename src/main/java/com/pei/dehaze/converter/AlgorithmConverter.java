package com.pei.dehaze.converter;

import com.pei.dehaze.model.entity.SysAlgorithm;
import com.pei.dehaze.model.vo.AlgorithmVO;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;

/**
 * @author earthy-zinc
 * @since 2024-06-09 00:11:21
 */
@Mapper(componentModel = "spring")
public interface AlgorithmConverter {
    
    @Mapping(ignore = true, target = "children")
    AlgorithmVO entity2Vo(SysAlgorithm entity);
}
