package com.pei.dehaze.model.read;

import lombok.Data;

import java.util.Set;

/**
 * 角色权限业务对象
 *
 * @author earthyzinc
 * @since 2023/11/29
 */
@Data
public class RolePermsRead {

    /**
     * 角色编码
     */
    private String roleCode;

    /**
     * 权限标识集合
     */
    private Set<String> perms;

}
