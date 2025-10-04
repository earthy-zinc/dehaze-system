package com.pei.dehaze.converter;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.model.entity.SysDict;
import com.pei.dehaze.model.form.DictForm;
import com.pei.dehaze.model.vo.DictPageVO;
import org.mapstruct.InheritInverseConfiguration;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;
import org.mapstruct.Mappings;

/**
 * 字典数据项对象转换器
 *
 * @author earthyzinc
 * @since 2022/6/8
 */
@Mapper(componentModel = "spring")
public interface DictConverter {
    @Mappings({
        @Mapping(ignore = true, target = "countId"),
        @Mapping(ignore = true, target = "maxLimit"),
        @Mapping(ignore = true, target = "optimizeCountSql"),
        @Mapping(ignore = true, target = "optimizeJoinOfCountSql"),
        @Mapping(ignore = true, target = "orders"),
        @Mapping(ignore = true, target = "searchCount"),
    })
    Page<DictPageVO> entity2Page(Page<SysDict> page);

    DictForm entity2Form(SysDict entity);

    @InheritInverseConfiguration(name="entity2Form")
    @Mapping(ignore = true, target = "defaulted")
    SysDict form2Entity(DictForm entity);
}
