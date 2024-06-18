package com.pei.dehaze.converter;

import com.pei.dehaze.model.entity.SysDataset;
import com.pei.dehaze.model.vo.DatasetVO;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;

/**
 * @author earthy-zinc
 * @since 2024-06-09 00:11:09
 */
@Mapper(componentModel = "spring")
public interface DatasetConverter {
    @Mapping(ignore = true, target = "children")
    DatasetVO entity2Vo(SysDataset entity);
}
