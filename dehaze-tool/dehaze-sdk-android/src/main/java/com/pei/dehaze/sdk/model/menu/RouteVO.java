package com.pei.dehaze.sdk.model.menu;

import lombok.Data;
import java.util.List;

/**
 * 路由对象模型类
 */
@Data
public class RouteVO {
    /**
     * 子路由列表
     */
    private List<RouteVO> children;
    
    /**
     * 组件路径
     */
    private String component;
    
    private Meta meta;
    
    /**
     * 路由名称
     */
    private String name;
    
    /**
     * 路由路径
     */
    private String path;
    
    /**
     * 跳转链接
     */
    private String redirect;
}