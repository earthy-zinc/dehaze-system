package com.pei.dehaze.converter;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.base.IBaseEnum;
import com.pei.dehaze.common.enums.GenderEnum;
import com.pei.dehaze.model.bo.UserBO;
import com.pei.dehaze.model.entity.SysUser;
import com.pei.dehaze.model.form.UserForm;
import com.pei.dehaze.model.vo.UserInfoVO;
import com.pei.dehaze.model.vo.UserPageVO;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;
import org.mapstruct.Mappings;

/**
 * 用户对象转换器
 *
 * @author earthyzinc
 * @since 2022/6/8
 */
@Mapper(componentModel = "spring", imports = {IBaseEnum.class, GenderEnum.class})
public interface UserConverter {

    @Mappings({
            @Mapping(target = "genderLabel", expression = "java(IBaseEnum.getLabelByValue(bo.getGender(), GenderEnum.class))")
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

    @Mappings({
            @Mapping(target = "id", ignore = true),
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
}
