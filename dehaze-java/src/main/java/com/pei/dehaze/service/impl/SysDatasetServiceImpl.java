package com.pei.dehaze.service.impl;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.constant.SystemConstants;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.model.Option;
import com.pei.dehaze.common.util.TreeDataUtils;
import com.pei.dehaze.converter.DatasetConverter;
import com.pei.dehaze.mapper.SysDatasetMapper;
import com.pei.dehaze.model.entity.SysDataset;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.model.form.DatasetAddForm;
import com.pei.dehaze.model.form.DatasetUpdateForm;
import com.pei.dehaze.model.query.DatasetQuery;
import com.pei.dehaze.model.dto.DatasetStatistics;
import com.pei.dehaze.model.vo.DatasetVO;
import com.pei.dehaze.service.DatasetOperationService;
import com.pei.dehaze.service.SysDatasetItemService;
import com.pei.dehaze.service.SysDatasetService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.context.annotation.Lazy;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.LinkedList;

/**
 * 数据集服务实现
 * 职责：处理Dataset（数据集）的基础CRUD操作
 * 注意：跨服务的复合操作（如级联删除）已迁移到 DatasetOperationService
 *
 * @author earthy-zinc
 * @since 2024-06-08 18:37:17
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class SysDatasetServiceImpl extends ServiceImpl<SysDatasetMapper, SysDataset> implements SysDatasetService {

    private final DatasetConverter datasetConverter;

    @Lazy
    private final DatasetOperationService datasetOperationService;

    @Lazy
    private final SysDatasetItemService sysDatasetItemService;

    @Value("${file.datasetPath}")
    private String datasetPath;

    @Override
    @Cacheable(value = "dataset:list", key = "#queryParams.keyword != null ? #queryParams.keyword : 'all'", unless = "#result == null || #result.isEmpty()")
    public List<DatasetVO> getList(DatasetQuery queryParams) {
        List<SysDataset> datasets = this.list(
                new LambdaQueryWrapper<SysDataset>()
                        .like(
                                CharSequenceUtil.isNotBlank(queryParams.getKeyword()),
                                SysDataset::getName,
                                queryParams.getKeyword()
                        ));

        List<Long> rootIds = TreeDataUtils.findRootIds(
                datasets,
                SysDataset::getId,
                SysDataset::getParentId
        );

        return rootIds.stream()
                .flatMap(rootId -> buildDatasetTree(rootId, datasets).stream())
                .toList();
    }

    @Override
    @CacheEvict(value = "dataset:list", allEntries = true)
    public DatasetVO addDataset(DatasetAddForm dataset) {
        // 校验同父节点下名称唯一性
        Long parentId = dataset.getParentId() != null ? dataset.getParentId() : SystemConstants.ROOT_NODE_ID;
        boolean exists = this.count(new LambdaQueryWrapper<SysDataset>()
                .eq(SysDataset::getName, dataset.getName())
                .eq(SysDataset::getParentId, parentId)) > 0;
        if (exists) {
            throw new BusinessException("同父节点下已存在相同名称的数据集");
        }

        SysDataset sysDataset = datasetConverter.form2Entity(dataset);
        if (this.save(sysDataset)) {
            // 清除父数据集的统计缓存（新增子数据集会影响父数据集的统计）
            if (dataset.getParentId() != null && !dataset.getParentId().equals(SystemConstants.ROOT_NODE_ID)) {
                evictDatasetStatsCache(dataset.getParentId());
            }
            return datasetConverter.entity2Vo(sysDataset, calculateStatistics(sysDataset.getId()));
        } else {
            throw new BusinessException("新增数据集失败");
        }
    }

    @Override
    @CacheEvict(value = "dataset:stats", key = "#id")
    public DatasetVO updateDataset(Long id, DatasetUpdateForm form) {
        // 校验ID合法性
        if (id == null || id <= 0) {
            throw new BusinessException("数据集ID无效");
        }

        // 获取当前数据集
        SysDataset currentDataset = this.getById(id);
        if (currentDataset == null) {
            throw new BusinessException("数据集不存在");
        }

        // 如果修改了名称，需要校验同父节点下名称唯一性
        if (form.getName() != null && !form.getName().equals(currentDataset.getName())) {
            boolean exists = this.count(new LambdaQueryWrapper<SysDataset>()
                    .eq(SysDataset::getName, form.getName())
                    .eq(SysDataset::getParentId, currentDataset.getParentId())
                    .ne(SysDataset::getId, id)) > 0;
            if (exists) {
                throw new BusinessException("同父节点下已存在相同名称的数据集");
            }
        }

        // 设置ID到entity
        SysDataset sysDataset = datasetConverter.updateForm2Entity(form);
        sysDataset.setId(id);

        if (this.updateById(sysDataset)) {
            return datasetConverter.entity2Vo(sysDataset, calculateStatistics(id));
        } else {
            throw new BusinessException("更新数据集失败");
        }
    }

    @Override
    @CacheEvict(value = {"dataset:list", "dataset:stats"}, key = "#id")
    public void deleteDataset(Long id) {
        // 校验ID合法性
        if (id == null || id <= 0) {
            throw new BusinessException("数据集ID无效");
        }

        // 检查数据集是否存在
        SysDataset dataset = this.getById(id);
        if (dataset == null) {
            throw new BusinessException("数据集不存在");
        }

        if (!this.removeById(id)) {
            throw new BusinessException("删除数据集失败");
        }
        // 清除已删除数据集的统计缓存
        evictDatasetStatsCache(id);
    }

    @Override
    public List<Option<Long>> getOptions() {
        List<SysDataset> datasets = this.list(new LambdaQueryWrapper<>());
        return buildDatasetOptions(SystemConstants.ROOT_NODE_ID, datasets);
    }

    /**
     * 获取所有叶子节点ID
     * 优化：一次性查询所有数据，在内存中处理，避免N+1查询
     */
    @Override
    public List<Long> getLeafDatasetIds() {
        // 一次性查询所有数据集
        List<SysDataset> allDatasets = this.list();
        if (allDatasets.isEmpty()) {
            return new ArrayList<>();
        }

        // 获取所有作为父节点的ID集合
        Set<Long> parentIds = allDatasets.stream()
                .filter(d -> d.getParentId() != null && !d.getParentId().equals(SystemConstants.ROOT_NODE_ID))
                .map(SysDataset::getParentId)
                .collect(java.util.stream.Collectors.toSet());

        // 过滤出不在parentIds中的ID，即为叶子节点
        return allDatasets.stream()
                .map(SysDataset::getId)
                .filter(id -> !parentIds.contains(id))
                .collect(java.util.stream.Collectors.toList());
    }

    /**
     * 获取当前节点的所有叶子节点id
     * 优化：一次性查询所有数据，在内存中BFS遍历，避免递归查库
     */
    @Override
    public List<Long> getLeafDatasetId(Long id) {
        // 一次性查询所有数据集
        List<SysDataset> allDatasets = this.list();
        if (allDatasets.isEmpty()) {
            return new ArrayList<>();
        }

        // 构建 parent -> children 映射
        Map<Long, List<SysDataset>> parentToChildrenMap = allDatasets.stream()
                .filter(d -> d.getParentId() != null)
                .collect(java.util.stream.Collectors.groupingBy(SysDataset::getParentId));

        List<Long> leafIds = new ArrayList<>();
        // 使用队列进行BFS遍历
        Queue<Long> queue = new LinkedList<>();
        queue.offer(id);

        while (!queue.isEmpty()) {
            Long currentId = queue.poll();
            List<SysDataset> children = parentToChildrenMap.get(currentId);

            if (children == null || children.isEmpty()) {
                // 没有子节点，说明是叶子节点
                leafIds.add(currentId);
            } else {
                // 有子节点，将子节点加入队列
                for (SysDataset child : children) {
                    queue.offer(child.getId());
                }
            }
        }

        return leafIds;
    }

    @Override
    public List<Long> getDatasetAndDescendantIds(Long datasetId) {
        List<Long> allIds = new ArrayList<>();
        allIds.add(datasetId);

        // 一次性查询所有数据集，避免重复查询
        List<SysDataset> allDatasets = this.list();
        if (allDatasets.isEmpty()) {
            return allIds;
        }

        // 构建 parent -> children 映射
        Map<Long, List<SysDataset>> parentToChildrenMap = allDatasets.stream()
                .filter(d -> d.getParentId() != null)
                .collect(java.util.stream.Collectors.groupingBy(SysDataset::getParentId));

        // 使用队列进行BFS遍历收集所有子孙ID
        Queue<Long> queue = new LinkedList<>();
        queue.offer(datasetId);

        while (!queue.isEmpty()) {
            Long currentId = queue.poll();
            List<SysDataset> children = parentToChildrenMap.get(currentId);

            if (children != null && !children.isEmpty()) {
                for (SysDataset child : children) {
                    allIds.add(child.getId());
                    queue.offer(child.getId());
                }
            }
        }

        return allIds;
    }

    @Override
    public SysDataset getRootDataset(Long id) {
        List<SysDataset> datasets = new ArrayList<>();
        // 获取当前节点
        SysDataset cur = this.getById(id);
        if (cur == null) {
            return null;
        }

        // 一次性查询所有数据集，避免递归查库
        List<SysDataset> allDatasets = this.list();
        // 构建id -> node的映射
        Map<Long, SysDataset> idToNodeMap = allDatasets.stream()
                .collect(java.util.stream.Collectors.toMap(SysDataset::getId, d -> d));

        // 执行深度优先遍历（内存中，不查库）
        dfsWithCache(cur, datasets, idToNodeMap);

        // 将子节点的 name 累加到父节点的 name 中
        if (!datasets.isEmpty()) {
            SysDataset root = datasets.get(datasets.size() - 1);
            // 将所有子节点的 name 追加到根节点的 name
            StringBuilder fullName = new StringBuilder(root.getName());
            StringBuilder fullDescription = new StringBuilder(root.getDescription());
            for (int i = datasets.size() - 2; i >= 0; i--) {
                SysDataset dataset = datasets.get(i);
                fullName.append("/").append(dataset.getName()); // 这里你可以自定义分隔符
                fullDescription.append("\n").append(dataset.getDescription());
            }
            root.setName(fullName.toString());
            root.setDescription(fullDescription.toString());
            return root;
        }
        return null;
    }

    @Override
    public SysDataset getSysDatasetById(Long id) {
        List<SysDataset> datasets = new ArrayList<>();
        // 获取当前节点
        SysDataset cur = this.getById(id);
        if (cur == null) {
            return null;
        }

        // 一次性查询所有数据集，避免递归查库
        List<SysDataset> allDatasets = this.list();
        // 构建id -> node的映射
        Map<Long, SysDataset> idToNodeMap = allDatasets.stream()
                .collect(java.util.stream.Collectors.toMap(SysDataset::getId, d -> d));

        // 执行深度优先遍历（内存中，不查库）
        dfsWithCache(cur, datasets, idToNodeMap);

        // 将祖先节点的 name 按照顺序追加到当前节点的 name 前面
        if (!datasets.isEmpty()) {
            StringBuilder fullName = new StringBuilder();
            StringBuilder fullDescription = new StringBuilder();
            // 从 root 到当前节点，依次追加每个节点的 name
            for (int i = datasets.size() - 1; i >= 0; i--) { // 从根节点到当前节点
                fullName.append(datasets.get(i).getName()).append("/");
                fullDescription.append(datasets.get(i).getDescription()).append("\n");
            }
            // 移除最后一个不必要的 "/"
            if (fullName.length() > 1) {
                fullName.setLength(fullName.length() - 1);
            }
            // 设置当前节点的 name
            cur.setName(fullName.toString());
            cur.setDescription(fullDescription.toString());
        }
        return cur;
    }

    /**
     * 深度优先遍历，获取从当前节点到根节点的路径（优化版：使用缓存，避免递归查库）
     *
     * @param cur         当前节点
     * @param datasets    路径结果列表
     * @param idToNodeMap ID到节点的映射缓存
     */
    private void dfsWithCache(SysDataset cur, List<SysDataset> datasets, Map<Long, SysDataset> idToNodeMap) {
        datasets.add(cur); // 将当前节点加入结果列表
        if (cur.getParentId() == null) {
            throw new BusinessException("数据集结构出现问题");
        }
        if (!cur.getParentId().equals(SystemConstants.ROOT_NODE_ID)) {
            // 从缓存中获取父节点，避免递归查库
            SysDataset parent = idToNodeMap.get(cur.getParentId());
            if (parent == null) {
                throw new BusinessException("数据集结构出现问题：无法找到父节点，parentId=" + cur.getParentId());
            }
            dfsWithCache(parent, datasets, idToNodeMap);
        }
    }

    private List<Option<Long>> buildDatasetOptions(Long rootNodeId, List<SysDataset> datasets) {
        List<Option<Long>> options = new ArrayList<>();
        for (SysDataset dataset : datasets) {
            if (dataset.getParentId().equals(rootNodeId)) {
                Option<Long> option = new Option<>(dataset.getId(), dataset.getName());
                List<Option<Long>> subDatasets = buildDatasetOptions(dataset.getId(), datasets);
                if (CollUtil.isNotEmpty(subDatasets)) {
                    option.setChildren(subDatasets);
                }
                options.add(option);
            }
        }
        return options;
    }

    private List<DatasetVO> buildDatasetTree(Long rootId, List<SysDataset> datasets) {
        return CollUtil.emptyIfNull(datasets)
                .stream()
                .filter(dataset -> dataset.getParentId().equals(rootId))
                .map(entity -> {
                    DatasetVO datasetVO = datasetConverter.entity2Vo(
                            entity,
                            calculateStatistics(entity.getId())
                    );
                    datasetVO.setChildren(buildDatasetTree(entity.getId(), datasets));
                    return datasetVO;
                }).toList();
    }

    @Override
    public DatasetVO getDatasetById(Long id) {
        // 校验ID合法性
        if (id == null || id <= 0) {
            throw new BusinessException("数据集ID无效");
        }

        SysDataset dataset = this.getSysDatasetById(id);
        if (dataset == null) {
            throw new BusinessException("数据集不存在");
        }

        DatasetStatistics stats = calculateStatistics(id);
        return datasetConverter.entity2Vo(dataset, stats);
    }

    /**
     * 计算数据集统计信息
     * 优化：将统计逻辑下沉到Mapper层，通过专用SQL直接聚合，减少内存占用
     * 注意：此方法为public以支持Spring AOP缓存代理
     */
    @Override
    @Cacheable(value = "dataset:stats", key = "#datasetId", unless = "#result == null")
    public DatasetStatistics calculateStatistics(Long datasetId) {
        DatasetStatistics stats = new DatasetStatistics();

        // 获取所有叶子节点ID
        List<Long> leafIds = this.getLeafDatasetId(datasetId);

        if (leafIds.isEmpty()) {
            stats.setItemCount(0L);
            stats.setFileCount(0L);
            stats.setTotalSize(0L);
            stats.setClearCount(0L);
            stats.setHazyCount(0L);
            stats.setSceneDistribution(new HashMap<>());
            stats.setHazeDistribution(new HashMap<>());
            stats.setFormatDistribution(new HashMap<>());
            return stats;
        }

        // 统计图片总数（通过Mapper SQL直接聚合）
        Long imageCount = this.baseMapper.countImagesByDatasetIds(leafIds);
        stats.setFileCount(imageCount != null ? imageCount : 0L);

        // 统计数据项总数
        Long itemCount = this.baseMapper.countItemsByDatasetIds(leafIds);
        stats.setItemCount(itemCount != null ? itemCount : 0L);

        // 统计文件总大小（字节）
        Long totalSize = this.baseMapper.countTotalSizeByDatasetIds(leafIds);
        stats.setTotalSize(totalSize != null ? totalSize : 0L);

        // 统计清晰图片数量
        Long clearCount = this.baseMapper.countClearImagesByDatasetIds(leafIds);
        stats.setClearCount(clearCount != null ? clearCount : 0L);

        // 统计有雾图片数量
        Long hazyCount = this.baseMapper.countHazyImagesByDatasetIds(leafIds);
        stats.setHazyCount(hazyCount != null ? hazyCount : 0L);

        if (imageCount == null || imageCount == 0) {
            stats.setSceneDistribution(new HashMap<>());
            stats.setHazeDistribution(new HashMap<>());
            stats.setFormatDistribution(new HashMap<>());
            return stats;
        }

        // 统计场景分布（通过Mapper SQL直接聚合）
        List<Map<String, Object>> sceneResults = this.baseMapper.countSceneDistribution(leafIds);
        Map<String, Long> sceneDistribution = convertToDistributionMap(sceneResults, "scene_type");
        stats.setSceneDistribution(sceneDistribution);

        // 统计雾霾程度分布（通过Mapper SQL直接聚合）
        List<Map<String, Object>> hazeResults = this.baseMapper.countHazeDistribution(leafIds);
        Map<String, Long> hazeDistribution = convertToDistributionMap(hazeResults, "haze_level");
        stats.setHazeDistribution(hazeDistribution);

        // 统计文件格式分布（通过Mapper SQL直接聚合）
        List<Map<String, Object>> formatResults = this.baseMapper.countFormatDistributionByDatasetIds(leafIds);
        Map<String, Long> formatDistribution = convertToDistributionMap(formatResults, "file_type");
        stats.setFormatDistribution(formatDistribution);

        return stats;
    }

    /**
     * 将查询结果转换为分布Map
     *
     * @param results  查询结果列表
     * @param keyField 键字段名
     * @return 分布Map
     */
    private Map<String, Long> convertToDistributionMap(List<Map<String, Object>> results,
                                                       String keyField) {
        Map<String, Long> distribution = new HashMap<>();
        if (results == null || results.isEmpty()) {
            return distribution;
        }

        for (Map<String, Object> row : results) {
            Object key = row.get(keyField);
            Object value = row.get("count");

            String keyStr = key != null ? key.toString() : "未知";
            long count = 0L;

            if (value instanceof Long) {
                count = (Long) value;
            } else if (value instanceof Integer) {
                count = ((Integer) value).longValue();
            } else if (value instanceof Number) {
                count = ((Number) value).longValue();
            }

            distribution.put(keyStr, count);
        }
        return distribution;
    }

    @Override
    public void incrementUsageCount(Long id) {
        this.baseMapper.incrementUsageCount(id);
    }

    @Override
    public List<SysItemFile> getDatasetImages(Long datasetId, boolean recursive) {
        if (recursive) {
            // 递归获取所有子数据集ID
            List<Long> datasetIds = this.getLeafDatasetId(datasetId);
            return baseMapper.getDatasetImages(datasetIds);
        } else {
            // 只获取当前数据集的图片
            return baseMapper.getDatasetImages(List.of(datasetId));
        }
    }

    @Override
    public String getDatasetNameByItemId(Long itemId) {
        // 参数校验
        if (itemId == null || itemId <= 0) {
            throw new BusinessException("数据项ID无效");
        }

        // 获取数据项
        SysDatasetItem datasetItem = sysDatasetItemService.getById(itemId);
        if (datasetItem == null) {
            throw new BusinessException("数据项不存在，itemId: " + itemId);
        }

        // 获取数据集
        SysDataset dataset = this.getById(datasetItem.getDatasetId());
        if (dataset == null) {
            log.warn("数据项关联的数据集不存在，itemId: {}, datasetId: {}", itemId, datasetItem.getDatasetId());
            return "";
        }

        return dataset.getName();
    }

    /**
     * 清除指定数据集的统计缓存
     */
    @CacheEvict(value = "dataset:stats", key = "#datasetId")
    public void evictDatasetStatsCache(Long datasetId) {
        log.debug("清除数据集统计缓存: datasetId={}", datasetId);
    }

    /**
     * 清除数据集及其所有祖先的统计缓存
     * 当子数据集发生变化时，需要清除所有祖先数据集的统计缓存
     */
    public void evictDatasetAndAncestorStatsCache(Long datasetId) {
        // 清除当前数据集的缓存
        evictDatasetStatsCache(datasetId);

        // 获取所有祖先数据集并清除缓存
        SysDataset dataset = this.getById(datasetId);
        if (dataset != null && dataset.getParentId() != null
                && !dataset.getParentId().equals(SystemConstants.ROOT_NODE_ID)) {
            evictDatasetAndAncestorStatsCache(dataset.getParentId());
        }
    }
}
