package com.pei.dehaze.common.util;

import java.util.List;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

public class TreeDataUtils {
    /**
     * 查找给定列表中所有元素的根元素ID。 根元素是指没有父元素的元素，或者可以说其父元素不在给定的元素列表中。 <br/>
     *
     * @param items          元素列表，这些元素可以通过idMapper和parentIdMapper函数映射到ID。
     * @param idMapper       函数用于映射元素到其ID。
     * @param parentIdMapper 函数用于映射元素到其父元素的ID。
     * @param <T>            元素的类型。
     * @param <ID>           元素和父元素ID的类型。
     * @return 包含所有根元素ID的列表。 <br/> 主要用于以下的都具有树结构的场景：
     * @see com.pei.dehaze.model.entity.SysDataset
     * @see com.pei.dehaze.model.entity.SysAlgorithm
     * @see com.pei.dehaze.model.entity.SysMenu
     */
    public static <T, ID> List<ID> findRootIds(List<T> items, Function<T, ID> idMapper, Function<T, ID> parentIdMapper) {
        // 收集所有元素的ID到一个集合中
        Set<ID> ids = items.stream()
                .map(idMapper)
                .collect(Collectors.toSet());

        // 收集所有元素父元素的ID到另一个集合中
        Set<ID> parentIds = items.stream()
                .map(parentIdMapper)
                .collect(Collectors.toSet());

        // 过滤出不在元素ID集合中的父元素ID，这些就是根元素的ID
        return parentIds.stream()
                .filter(id -> !ids.contains(id))
                .toList();
    }
}
