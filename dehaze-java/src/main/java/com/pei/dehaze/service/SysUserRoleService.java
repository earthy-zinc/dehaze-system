package com.pei.dehaze.service;


import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.entity.SysUserRole;

import java.util.List;

public interface SysUserRoleService extends IService<SysUserRole> {

    /**
     * 保存用户角色
     *
     * @param userId
     * @param roleIds
     * @return
     */
    boolean saveUserRoles(Long userId, List<Long> roleIds);

    /**
     * 判断角色是否存在绑定的用户
     *
     * @param roleId 角色ID
     * @return true：已分配 false：未分配
     */
    boolean hasAssignedUsers(Long roleId);

    /**
     * 批量查询角色关联的活跃用户名（软删用户不参与权限传播）
     *
     * @param roleIds 角色ID集合
     * @return 用户名列表（去重）
     */
    List<String> listUsernamesByRoleIds(List<Long> roleIds);
}
