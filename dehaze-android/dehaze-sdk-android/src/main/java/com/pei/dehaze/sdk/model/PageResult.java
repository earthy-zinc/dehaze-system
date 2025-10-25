package com.pei.dehaze.sdk.model;

import lombok.Data;
import java.util.List;

/**
 * 分页结果基类
 */
@Data
public class PageResult<T> {
    /**
     * 数据列表
     */
    private List<T> list;
    
    /**
     * 总记录数
     */
    private long total;
}