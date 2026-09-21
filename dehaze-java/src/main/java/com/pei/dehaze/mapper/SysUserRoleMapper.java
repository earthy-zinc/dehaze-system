package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysUserRole;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/**
 * 用户角色访问层
 *
 * @author earthyzinc
 * @since 2022/1/15
 */
@Mapper
public interface SysUserRoleMapper extends BaseMapper<SysUserRole> {

    /**
     * 获取角色绑定的用户数
     *
     * @param roleId 角色ID
     */
    long countUsersForRole(Long roleId);

    /**
     * 批量查询角色关联的活跃用户名（去重，软删用户不参与）
     *
     * @param roleIds 角色ID集合
     */
    List<String> listUsernamesByRoleIds(@Param("roleIds") List<Long> roleIds);
}
