package com.pei.dehaze.common.constant;

/**
 * 系统常量
 *
 * @author earthyzinc
 * @since 1.0.0
 */
public class SystemConstants {

    /**
     * 根节点ID
     */
    public static final Long ROOT_NODE_ID = 0L;

    /**
     * 超级管理员角色编码
     */
    public static final String ROOT_ROLE_CODE = "ROOT";

    /** 内置角色编码集合（ROOT/ADMIN，禁止删除与状态修改） */
    public static final java.util.Set<String> BUILTIN_ROLE_CODES = java.util.Set.of(ROOT_ROLE_CODE, "ADMIN");

    /**
     * 管理员角色编码
     */
    public static final String ADMIN_ROLE_CODE = "ADMIN";

    /**
     * 系统用户ID
     */
    public static final Long SYSTEM_USER_ID = 0L;

    /**
     * 系统用户名
     */
    public static final String SYSTEM_USERNAME = "system";

    /**
     * 角色选项缓存 Key（TTL 1h）
     */
    public static final String ROLE_OPTIONS_CACHE_KEY = "role:options";
}
