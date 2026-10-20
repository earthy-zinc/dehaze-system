package com.pei.clarity.converter;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.clarity.common.base.IBaseEnum;
import com.pei.clarity.common.enums.GenderEnum;
import com.pei.clarity.model.entity.SysUser;
import com.pei.clarity.model.form.UserForm;
import com.pei.clarity.model.read.UserRead;
import com.pei.clarity.model.vo.UserInfoVO;
import com.pei.clarity.model.vo.UserPageVO;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;
import org.mapstruct.Mappings;

/**
 * 用户对象转换器
 *
 * @author earthyzinc
 * @since 2022/6/8
 */
@Mapper(
        componentModel = "spring",
        imports = {IBaseEnum.class, GenderEnum.class})
public interface UserConverter {

    @Mappings({
        @Mapping(
                target = "genderLabel",
                expression = "java(IBaseEnum.getLabelByValue(read.getGender(), GenderEnum.class))")
    })
    UserPageVO read2PageVo(UserRead read);

    @Mappings({
        @Mapping(ignore = true, target = "countId"),
        @Mapping(ignore = true, target = "maxLimit"),
        @Mapping(ignore = true, target = "optimizeCountSql"),
        @Mapping(ignore = true, target = "optimizeJoinOfCountSql"),
        @Mapping(ignore = true, target = "orders"),
        @Mapping(ignore = true, target = "searchCount"),
    })
    Page<UserPageVO> read2PageVo(Page<UserRead> read);

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
