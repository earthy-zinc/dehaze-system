package com.pei.clarity.converter;

import com.pei.clarity.model.entity.SysMenu;
import com.pei.clarity.model.form.MenuForm;
import com.pei.clarity.model.vo.MenuVO;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;
import org.mapstruct.Mappings;

/**
 * 菜单对象转换器
 *
 * @author earthyzinc
 * @since 2022/7/29
 */
@Mapper(componentModel = "spring")
public interface MenuConverter {
    @Mapping(ignore = true, target = "children")
    MenuVO entity2Vo(SysMenu entity);

    MenuForm entity2Form(SysMenu entity);

    @Mappings({
        @Mapping(target = "createTime", ignore = true),
        @Mapping(target = "updateTime", ignore = true),
        @Mapping(target = "treePath", ignore = true),
    })
    SysMenu form2Entity(MenuForm menuForm);
}
