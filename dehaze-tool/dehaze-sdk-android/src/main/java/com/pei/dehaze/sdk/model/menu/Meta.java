package com.pei.dehaze.sdk.model.menu;

import lombok.Data;
import java.util.List;

/**
 * 路由属性模型类
 */
@Data
public class Meta {
    /**
     * 【目录】只有一个子路由是否始终显示
     */
    private Boolean alwaysShow;
    
    /**
     * 是否隐藏(true-是 false-否)
     */
    private Boolean hidden;
    
    /**
     * ICON
     */
    private String icon;
    
    /**
     * 【菜单】是否开启页面缓存
     */
    private Boolean keepAlive;
    
    /**
     * 拥有路由权限的角色编码
     */
    private List<String> roles;
    
    /**
     * 路由title
     */
    private String title;
}