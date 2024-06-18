package com.pei.dehaze.converter;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.model.Option;
import com.pei.dehaze.model.entity.SysRole;
import com.pei.dehaze.model.form.RoleForm;
import com.pei.dehaze.model.vo.RolePageVO;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;
import org.mapstruct.Mappings;

import java.util.List;

/**
 * 角色对象转换器
 *
 * @author earthyzinc
 * @since 2022/5/29
 */
@Mapper(componentModel = "spring")
public interface RoleConverter {
    @Mappings({
        @Mapping(ignore = true, target = "countId"),
        @Mapping(ignore = true, target = "maxLimit"),
        @Mapping(ignore = true, target = "optimizeCountSql"),
        @Mapping(ignore = true, target = "optimizeJoinOfCountSql"),
        @Mapping(ignore = true, target = "orders"),
        @Mapping(ignore = true, target = "searchCount"),
    })
    Page<RolePageVO> entity2Page(Page<SysRole> page);

    @Mappings({
            @Mapping(target = "value", source = "id"),
            @Mapping(target = "label", source = "name"),
            @Mapping(ignore = true, target = "children")
    })
    Option<Long> entity2Option(SysRole role);


    List<Option<Long>> entities2Options(List<SysRole> roles);

    @Mappings({
            @Mapping(target = "createTime", ignore = true),
            @Mapping(target = "updateTime", ignore = true),
            @Mapping(target = "deleted", ignore = true),
    })
    SysRole form2Entity(RoleForm roleForm);

    RoleForm entity2Form(SysRole entity);
}
