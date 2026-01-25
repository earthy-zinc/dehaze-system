package com.pei.dehaze.converter;

import com.pei.dehaze.model.entity.SysAlgorithm;
import com.pei.dehaze.model.form.AlgorithmForm;
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
    @Mapping(ignore = true, target = "size")
    @Mapping(ignore = true, target = "img")
    @Mapping(ignore = true, target = "params")
    @Mapping(ignore = true, target = "flops")
    AlgorithmVO entity2Vo(SysAlgorithm entity);

    @Mapping(ignore = true, target = "id")
    @Mapping(ignore = true, target = "parentId")
    @Mapping(ignore = true, target = "createTime")
    @Mapping(ignore = true, target = "updateTime")
    @Mapping(ignore = true, target = "createBy")
    @Mapping(ignore = true, target = "updateBy")
    @Mapping(ignore = true, target = "size")
    @Mapping(ignore = true, target = "img")
    @Mapping(ignore = true, target = "params")
    @Mapping(ignore = true, target = "flops")
    SysAlgorithm form2Entity(AlgorithmForm form);
}
