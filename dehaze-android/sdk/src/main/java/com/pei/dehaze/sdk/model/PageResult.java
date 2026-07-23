package com.pei.dehaze.sdk.model;

import lombok.Data;
import org.jetbrains.annotations.NotNull;

import java.util.Collections;
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

    /**
     * 获取数据列表，保证非 null（Gson 反序列化缺失字段时返回空列表）
     */
    @NotNull
    public List<T> getList() {
        return list != null ? list : Collections.emptyList();
    }
}
