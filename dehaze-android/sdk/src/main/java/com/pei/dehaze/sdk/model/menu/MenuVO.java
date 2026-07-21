package com.pei.dehaze.sdk.model.menu;

import lombok.Data;
import java.util.List;

/**
 * 菜单视图对象模型类
 */
@Data
public class MenuVO {
    /**
     * 子菜单
     */
    private List<MenuVO> children;
    
    /**
     * 组件路径
     */
    private String component;
    
    /**
     * ICON
     */
    private String icon;
    
    /**
     * 菜单ID
     */
    private Integer id;
    
    /**
     * 菜单名称
     */
    private String name;
    
    /**
     * 父菜单ID
     */
    private Integer parentId;
    
    /**
     * 按钮权限标识
     */
    private String perm;
    
    /**
     * 跳转路径
     */
    private String redirect;

    /**
     * 路由路径
     */
    private String path;
    
    /**
     * 菜单排序(数字越小排名越靠前)
     */
    private Integer sort;
    
    /**
     * 菜单类型(CATALOG:目录;MENU:菜单;BUTTON:按钮;EXTLINK:外链)
     */
    private String type;
    
    /**
     * 菜单是否可见(1:显示;0:隐藏)
     */
    private Integer visible;
}