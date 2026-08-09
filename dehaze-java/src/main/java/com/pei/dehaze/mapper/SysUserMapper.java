package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.model.read.UserRead;
import com.pei.dehaze.model.dto.UserAuthInfo;
import com.pei.dehaze.model.entity.SysUser;
import com.pei.dehaze.model.form.UserForm;
import com.pei.dehaze.model.query.UserPageQuery;
import com.pei.dehaze.plugin.mybatis.annotation.DataPermission;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

/**
 * 用户持久层
 *
 * @author earthyzinc
 * @since 2022/1/14
 */
@Mapper
public interface SysUserMapper extends BaseMapper<SysUser> {

    /**
     * 获取用户分页列表
     *
     * @param page
     * @param queryParams 查询参数
     * @return
     */
    @DataPermission(deptAlias = "u")
    Page<UserRead> listPagedUsers(Page<UserRead> page, UserPageQuery queryParams);

    /**
     * 获取用户表单详情
     *
     * @param userId 用户ID
     * @return
     */
    UserForm getUserFormData(Long userId);

    /**
     * 根据用户名获取认证信息
     *
     * @param username
     * @return
     */
    UserAuthInfo getUserAuthInfo(String username);

    /**
     * 按用户名在全表范围（含已软删行）查重。
     * MyBatis-Plus @TableLogic 会自动追加 deleted=0，此处必须用原生 SQL 绕过。
     *
     * @param username 用户名
     * @return 命中行数
     */
    @Select("SELECT COUNT(*) FROM sys_user WHERE username = #{username}")
    long countByUsernameAllDeleted(String username);
}
