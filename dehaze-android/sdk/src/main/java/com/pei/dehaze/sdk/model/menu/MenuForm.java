package com.pei.dehaze.sdk.model.menu;

import lombok.Data;

/**
 * 菜单表单对象模型类
 */
@Data
public class MenuForm {
    /**
     * 菜单ID
     */
    private String id;

    /**
     * 父菜单ID
     */
    private Integer parentId;

    /**
     * 菜单名称
     */
    private String name;

    /**
     * 菜单是否可见(1:是;0:否;)
     */
    private int visible;

    private String icon;

    /**
     * 排序
     */
    private int sort;

    /**
     * 组件路径
     */
    private String component;

    /**
     * 路由路径
     */
    private String path;

    /**
     * 跳转路由路径
     */
    private String redirect;

    /**
     * 菜单类型(CATALOG:目录;MENU:菜单;BUTTON:按钮;EXTLINK:外链)
     */
    private String type;

    /**
     * 权限标识
     */
    private String perm;

    /**
     * 【菜单】是否开启页面缓存
     */
    private Integer keepAlive;

    /**
     * 【目录】只有一个子路由是否始终显示
     */
    private Integer alwaysShow;
}
