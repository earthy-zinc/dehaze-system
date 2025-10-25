package com.pei.dehaze.sdk.model;

import lombok.Data;

/**
 * 分页查询参数基类
 */
@Data
public class PageQuery {
    /**
     * 页码
     */
    private int pageNum = 1;
    
    /**
     * 每页大小
     */
    private int pageSize = 10;
}