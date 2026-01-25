package com.pei.dehaze.common.util;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * 树形数据工具类
 * 提供树结构数据的通用计算方法
 *
 * @author earthy-zinc
 */
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

    /**
     * 查找所有叶子节点ID（没有子节点的节点）
     *
     * @param items          元素列表
     * @param idMapper       函数用于映射元素到其ID
     * @param parentIdMapper 函数用于映射元素到其父元素的ID
     * @param <T>            元素的类型
     * @param <ID>           元素ID的类型
     * @return 所有叶子节点ID列表
     */
    public static <T, ID> List<ID> findAllLeafIds(List<T> items, Function<T, ID> idMapper, Function<T, ID> parentIdMapper) {
        if (items == null || items.isEmpty()) {
            return Collections.emptyList();
        }

        // 获取所有作为父节点的ID集合
        Set<ID> parentIds = items.stream()
                .map(parentIdMapper)
                .filter(Objects::nonNull)
                .collect(Collectors.toSet());

        // 过滤出不在parentIds中的ID，即为叶子节点
        return items.stream()
                .map(idMapper)
                .filter(id -> !parentIds.contains(id))
                .collect(Collectors.toList());
    }

    /**
     * 查找指定节点下的所有叶子节点ID（BFS遍历）
     *
     * @param items          元素列表
     * @param rootId         起始节点ID
     * @param idMapper       函数用于映射元素到其ID
     * @param parentIdMapper 函数用于映射元素到其父元素的ID
     * @param <T>            元素的类型
     * @param <ID>           元素ID的类型
     * @return 指定节点下的所有叶子节点ID列表
     */
    public static <T, ID> List<ID> findLeafIdsUnder(List<T> items, ID rootId,
                                                    Function<T, ID> idMapper,
                                                    Function<T, ID> parentIdMapper) {
        if (items == null || items.isEmpty() || rootId == null) {
            return Collections.emptyList();
        }

        // 构建 parent -> children 映射
        Map<ID, List<T>> parentToChildrenMap = items.stream()
                .filter(item -> parentIdMapper.apply(item) != null)
                .collect(Collectors.groupingBy(parentIdMapper));

        List<ID> leafIds = new ArrayList<>();
        // 使用队列进行BFS遍历
        Queue<ID> queue = new LinkedList<>();
        queue.offer(rootId);

        while (!queue.isEmpty()) {
            ID currentId = queue.poll();
            List<T> children = parentToChildrenMap.get(currentId);

            if (children == null || children.isEmpty()) {
                // 没有子节点，说明是叶子节点
                leafIds.add(currentId);
            } else {
                // 有子节点，将子节点加入队列
                for (T child : children) {
                    queue.offer(idMapper.apply(child));
                }
            }
        }

        return leafIds;
    }

    /**
     * 查找指定节点及其所有子孙节点ID
     *
     * @param items          元素列表
     * @param rootId         起始节点ID
     * @param idMapper       函数用于映射元素到其ID
     * @param parentIdMapper 函数用于映射元素到其父元素的ID
     * @param <T>            元素的类型
     * @param <ID>           元素ID的类型
     * @return 指定节点及其所有子孙节点ID列表
     */
    public static <T, ID> List<ID> findDescendantIds(List<T> items, ID rootId,
                                                     Function<T, ID> idMapper,
                                                     Function<T, ID> parentIdMapper) {
        if (items == null || items.isEmpty() || rootId == null) {
            return Collections.emptyList();
        }

        List<ID> allIds = new ArrayList<>();
        allIds.add(rootId);

        // 构建 parent -> children 映射
        Map<ID, List<T>> parentToChildrenMap = items.stream()
                .filter(item -> parentIdMapper.apply(item) != null)
                .collect(Collectors.groupingBy(parentIdMapper));

        // 使用队列进行BFS遍历
        Queue<ID> queue = new LinkedList<>();
        queue.offer(rootId);

        while (!queue.isEmpty()) {
            ID currentId = queue.poll();
            List<T> children = parentToChildrenMap.get(currentId);

            if (children != null && !children.isEmpty()) {
                for (T child : children) {
                    ID childId = idMapper.apply(child);
                    allIds.add(childId);
                    queue.offer(childId);
                }
            }
        }

        return allIds;
    }
}
