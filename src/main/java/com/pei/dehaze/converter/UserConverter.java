package com.pei.dehaze.converter;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.model.bo.UserBO;
import com.pei.dehaze.model.entity.SysUser;
import com.pei.dehaze.model.form.UserForm;
import com.pei.dehaze.model.vo.UserImportVO;
import com.pei.dehaze.model.vo.UserInfoVO;
import com.pei.dehaze.model.vo.UserPageVO;
import org.mapstruct.InheritInverseConfiguration;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;
import org.mapstruct.Mappings;

/**
 * 用户对象转换器
 *
 * @author earthyzinc
 * @since 2022/6/8
 */
@Mapper(componentModel = "spring")
public interface UserConverter {

    @Mappings({
            @Mapping(target = "genderLabel", expression = "java(com.pei.dehaze.common.base.IBaseEnum.getLabelByValue(bo.getGender(), com.pei.dehaze.common.enums.GenderEnum.class))")
    })
    UserPageVO bo2PageVo(UserBO bo);

    @Mappings({
        @Mapping(ignore = true, target = "countId"),
        @Mapping(ignore = true, target = "maxLimit"),
        @Mapping(ignore = true, target = "optimizeCountSql"),
        @Mapping(ignore = true, target = "optimizeJoinOfCountSql"),
        @Mapping(ignore = true, target = "orders"),
        @Mapping(ignore = true, target = "searchCount"),
    })
    Page<UserPageVO> bo2PageVo(Page<UserBO> bo);

    @InheritInverseConfiguration(name = "entity2Form")
    @Mappings({
                    @Mapping(target = "createTime", ignore = true),
                    @Mapping(target = "updateTime", ignore = true),
                    @Mapping(target = "deleted", ignore = true),
                    @Mapping(target = "password", ignore = true),
    })
    SysUser form2Entity(UserForm entity);

    @Mappings({
            @Mapping(target = "userId", source = "id"),
            @Mapping(target = "roles", ignore = true),
            @Mapping(target = "perms", ignore = true),
    })
    UserInfoVO toUserInfoVo(SysUser entity);
    
    @Mappings({
            @Mapping(target = "createTime", ignore = true),
            @Mapping(target = "updateTime", ignore = true),
            @Mapping(target = "avatar", ignore = true),
            @Mapping(target = "deleted", ignore = true),
            @Mapping(target = "deptId", ignore = true),
            @Mapping(target = "gender", ignore = true),
            @Mapping(target = "id", ignore = true),
            @Mapping(target = "password", ignore = true),
            @Mapping(target = "status", ignore = true),
    })
    SysUser importVo2Entity(UserImportVO vo);
}
