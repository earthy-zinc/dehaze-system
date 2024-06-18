package com.pei.dehaze.converter;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.model.entity.SysDictType;
import com.pei.dehaze.model.form.DictTypeForm;
import com.pei.dehaze.model.vo.DictTypePageVO;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;
import org.mapstruct.Mappings;

/**
 * 字典类型对象转换器
 *
 * @author earthyzinc
 * @since 2022/6/8
 */
@Mapper(componentModel = "spring")
public interface DictTypeConverter {
    @Mappings({
            @Mapping(ignore = true, target = "countId"),
            @Mapping(ignore = true, target = "maxLimit"),
            @Mapping(ignore = true, target = "optimizeCountSql"),
            @Mapping(ignore = true, target = "optimizeJoinOfCountSql"),
            @Mapping(ignore = true, target = "orders"),
            @Mapping(ignore = true, target = "searchCount"),
    })
    Page<DictTypePageVO> entity2Page(Page<SysDictType> page);

    DictTypeForm entity2Form(SysDictType entity);

    @Mappings({
        @Mapping(target = "createTime", ignore = true),
        @Mapping(target = "updateTime", ignore = true),
    })
    SysDictType form2Entity(DictTypeForm entity);
}
