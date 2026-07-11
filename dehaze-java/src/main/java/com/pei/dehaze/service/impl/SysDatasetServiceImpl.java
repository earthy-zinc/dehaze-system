package com.pei.dehaze.service.impl;

import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.constant.SystemConstants;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.model.Option;
import com.pei.dehaze.common.util.TreeDataUtils;
import com.pei.dehaze.converter.DatasetConverter;
import com.pei.dehaze.mapper.SysDatasetMapper;
import com.pei.dehaze.model.entity.SysDataset;
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

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.stream.Collectors;

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
    @Cacheable(value = "dataset:all", key = "'all'", unless = "#result == null || #result.isEmpty()")
    public List<SysDataset> getAllDatasets() {
        return this.list(new LambdaQueryWrapper<SysDataset>()
                .orderByAsc(SysDataset::getId));
    }

    @Override
    @Cacheable(value = "dataset:statsMap", key = "'all'", unless = "#result == null || #result.isEmpty()")
    public Map<Long, DatasetStatistics> getAllDatasetStats() {
        long startTime = System.currentTimeMillis();
        log.debug("开始计算所有数据集统计信息...");

        List<SysDataset> allDatasets = getAllDatasets();
        Map<Long, DatasetStatistics> statsMap = new HashMap<>();

        if (allDatasets.isEmpty()) {
            return statsMap;
        }

        for (SysDataset ds : allDatasets) {
            DatasetStatistics empty = createEmptyStats();
            statsMap.put(ds.getId(), empty);
        }

        Set<Long> allChildParentIds = allDatasets.stream()
                .filter(d -> d.getParentId() != null && !d.getParentId().equals(SystemConstants.ROOT_NODE_ID))
                .map(SysDataset::getParentId)
                .collect(Collectors.toSet());

        List<Long> leafIds = allDatasets.stream()
                .map(SysDataset::getId)
                .filter(id -> !allChildParentIds.contains(id))
                .toList();

        if (leafIds.isEmpty()) {
            log.debug("没有叶子数据集，返回空统计");
            return statsMap;
        }

        log.debug("发现 {} 个叶子数据集，开始批量查询统计信息", leafIds.size());

        List<Map<String, Object>> itemResults = this.baseMapper.countItemsPerDataset(leafIds);
        for (Map<String, Object> row : itemResults) {
            Long dsId = ((Number) row.get("dataset_id")).longValue();
            if (statsMap.containsKey(dsId)) {
                statsMap.get(dsId).setItemCount(((Number) row.get("cnt")).longValue());
            }
        }

        List<Map<String, Object>> statsResults = this.baseMapper.countDatasetStatsBatch(leafIds);
        for (Map<String, Object> row : statsResults) {
            Long dsId = ((Number) row.get("dataset_id")).longValue();
            if (!statsMap.containsKey(dsId)) {
                continue;
            }
            DatasetStatistics stats = statsMap.get(dsId);
            Object imgCnt = row.get("image_count");
            stats.setFileCount(imgCnt != null ? ((Number) imgCnt).longValue() : 0L);
            Object sizeVal = row.get("total_size");
            stats.setTotalSize(sizeVal != null ? ((Number) sizeVal).longValue() : 0L);
            Object clearCnt = row.get("clear_count");
            stats.setClearCount(clearCnt != null ? ((Number) clearCnt).longValue() : 0L);
            Object hazyCnt = row.get("hazy_count");
            stats.setHazyCount(hazyCnt != null ? ((Number) hazyCnt).longValue() : 0L);
        }

        List<Map<String, Object>> sceneResults = this.baseMapper.countSceneDistributionBatch(leafIds);
        for (Map<String, Object> row : sceneResults) {
            Long dsId = ((Number) row.get("dataset_id")).longValue();
            if (!statsMap.containsKey(dsId)) continue;
            String key = String.valueOf(row.get("scene_type"));
            long cnt = row.get("cnt") instanceof Number ? ((Number) row.get("cnt")).longValue() : 0L;
            statsMap.get(dsId).getSceneDistribution().merge(key, cnt, Long::sum);
        }

        List<Map<String, Object>> hazeResults = this.baseMapper.countHazeDistributionBatch(leafIds);
        for (Map<String, Object> row : hazeResults) {
            Long dsId = ((Number) row.get("dataset_id")).longValue();
            if (!statsMap.containsKey(dsId)) continue;
            String key = String.valueOf(row.get("haze_level"));
            long cnt = row.get("cnt") instanceof Number ? ((Number) row.get("cnt")).longValue() : 0L;
            statsMap.get(dsId).getHazeDistribution().merge(key, cnt, Long::sum);
        }

        List<Map<String, Object>> formatResults = this.baseMapper.countFormatDistributionBatch(leafIds);
        for (Map<String, Object> row : formatResults) {
            Long dsId = ((Number) row.get("dataset_id")).longValue();
            if (!statsMap.containsKey(dsId)) continue;
            String key = String.valueOf(row.get("file_type"));
            long cnt = row.get("cnt") instanceof Number ? ((Number) row.get("cnt")).longValue() : 0L;
            statsMap.get(dsId).getFormatDistribution().merge(key, cnt, Long::sum);
        }

        Map<Long, List<Long>> parentToChildrenMap = new HashMap<>();
        for (SysDataset ds : allDatasets) {
            if (ds.getParentId() != null && !ds.getParentId().equals(SystemConstants.ROOT_NODE_ID)) {
                parentToChildrenMap.computeIfAbsent(ds.getParentId(), k -> new ArrayList<>()).add(ds.getId());
            }
        }

        Queue<Long> queue = new LinkedList<>(leafIds);
        Set<Long> processed = new java.util.HashSet<>(leafIds);

        while (!queue.isEmpty()) {
            Long currentId = queue.poll();
            SysDataset current = allDatasets.stream()
                    .filter(d -> d.getId().equals(currentId))
                    .findFirst()
                    .orElse(null);
            if (current == null || current.getParentId() == null
                    || current.getParentId().equals(SystemConstants.ROOT_NODE_ID)) {
                continue;
            }
            Long parentId = current.getParentId();
            DatasetStatistics parentStats = statsMap.get(parentId);
            DatasetStatistics childStats = statsMap.get(currentId);
            if (parentStats != null && childStats != null) {
                mergeStats(parentStats, childStats);
            }

            List<Long> siblings = parentToChildrenMap.getOrDefault(parentId, Collections.emptyList());
            boolean allSiblingsProcessed = siblings.stream().allMatch(processed::contains);
            if (allSiblingsProcessed && processed.add(parentId)) {
                queue.offer(parentId);
            }
        }

        long costMs = System.currentTimeMillis() - startTime;
        log.info("所有数据集统计信息计算完成，耗时 {} ms，叶子节点 {} 个", costMs, leafIds.size());
        return statsMap;
    }

    private DatasetStatistics createEmptyStats() {
        DatasetStatistics stats = new DatasetStatistics();
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

    private void mergeStats(DatasetStatistics parent, DatasetStatistics child) {
        parent.setItemCount(parent.getItemCount() + child.getItemCount());
        parent.setFileCount(parent.getFileCount() + child.getFileCount());
        parent.setTotalSize(parent.getTotalSize() + child.getTotalSize());
        parent.setClearCount(parent.getClearCount() + child.getClearCount());
        parent.setHazyCount(parent.getHazyCount() + child.getHazyCount());
        child.getSceneDistribution().forEach((k, v) ->
                parent.getSceneDistribution().merge(k, v, Long::sum));
        child.getHazeDistribution().forEach((k, v) ->
                parent.getHazeDistribution().merge(k, v, Long::sum));
        child.getFormatDistribution().forEach((k, v) ->
                parent.getFormatDistribution().merge(k, v, Long::sum));
    }

    @CacheEvict(value = {"dataset:all", "dataset:options", "dataset:page", "dataset:children", "dataset:detail", "dataset:stats", "dataset:statsMap"}, allEntries = true)
    public void evictAllDatasetsCache() {
        log.debug("清除所有数据集缓存");
    }

    @Override
    public IPage<DatasetVO> listPagedDatasets(DatasetQuery queryParams) {
        int pageNum = queryParams.getPageNum();
        int pageSize = queryParams.getPageSize();
        String keyword = queryParams.getKeyword();
        String type = queryParams.getType();
        Integer status = queryParams.getStatus();

        LambdaQueryWrapper<SysDataset> wrapper = new LambdaQueryWrapper<SysDataset>()
                .eq(SysDataset::getParentId, SystemConstants.ROOT_NODE_ID)
                .like(CharSequenceUtil.isNotBlank(keyword), SysDataset::getName, keyword)
                .eq(CharSequenceUtil.isNotBlank(type), SysDataset::getType, type)
                .eq(status != null, SysDataset::getStatus, status)
                .orderByAsc(SysDataset::getId);

        Page<SysDataset> page = this.page(new Page<>(pageNum, pageSize), wrapper);

        List<SysDataset> rootDatasets = page.getRecords();
        if (rootDatasets.isEmpty()) {
            return new Page<DatasetVO>(pageNum, pageSize, page.getTotal())
                    .setRecords(Collections.emptyList());
        }

        List<Long> rootIds = rootDatasets.stream().map(SysDataset::getId).toList();

        List<SysDataset> children = this.list(new LambdaQueryWrapper<SysDataset>()
                .in(SysDataset::getParentId, rootIds)
                .orderByAsc(SysDataset::getId));

        Set<Long> parentIds = children.stream()
                .map(SysDataset::getParentId)
                .collect(Collectors.toSet());

        Map<Long, List<SysDataset>> directChildrenMap = children.stream()
                .collect(Collectors.groupingBy(SysDataset::getParentId));

        List<Long> childIds = children.stream().map(SysDataset::getId).toList();
        List<SysDataset> grandChildren = this.list(new LambdaQueryWrapper<SysDataset>()
                .in(SysDataset::getParentId, childIds)
                .select(SysDataset::getParentId));
        Set<Long> grandParentIds = grandChildren.stream()
                .map(SysDataset::getParentId)
                .collect(Collectors.toSet());

        Map<Long, Boolean> hasChildrenMap = new HashMap<>();
        for (Long rootId : rootIds) {
            hasChildrenMap.put(rootId, directChildrenMap.containsKey(rootId) && !directChildrenMap.get(rootId).isEmpty());
        }
        for (SysDataset child : children) {
            hasChildrenMap.put(child.getId(), grandParentIds.contains(child.getId()));
        }

        Map<Long, DatasetStatistics> statsMap = getAllDatasetStats();

        List<DatasetVO> records = rootDatasets.stream()
                .map(entity -> {
                    DatasetStatistics stats = statsMap.get(entity.getId());
                    DatasetVO vo = datasetConverter.entity2Vo(entity, stats);
                    vo.setHasChildren(hasChildrenMap.getOrDefault(entity.getId(), false));
                    vo.setTotal(stats != null ? stats.getFileCount() : 0L);
                    List<DatasetVO> childVOs = directChildrenMap.getOrDefault(entity.getId(), Collections.emptyList())
                            .stream()
                            .map(child -> {
                                DatasetStatistics childStats = statsMap.get(child.getId());
                                DatasetVO childVo = datasetConverter.entity2Vo(child, childStats);
                                childVo.setHasChildren(hasChildrenMap.getOrDefault(child.getId(), false));
                                childVo.setTotal(childStats != null ? childStats.getFileCount() : 0L);
                                return childVo;
                            })
                            .toList();
                    vo.setChildren(childVOs);
                    return vo;
                })
                .toList();

        Page<DatasetVO> resultPage = new Page<>(pageNum, pageSize, page.getTotal());
        resultPage.setRecords(records);
        return resultPage;
    }

    @Override
    public List<DatasetVO> listChildren(Long parentId) {
        if (parentId == null || parentId <= 0) {
            return Collections.emptyList();
        }

        List<SysDataset> children = this.list(new LambdaQueryWrapper<SysDataset>()
                .eq(SysDataset::getParentId, parentId)
                .orderByAsc(SysDataset::getId));

        if (children.isEmpty()) {
            return Collections.emptyList();
        }

        List<Long> childIds = children.stream().map(SysDataset::getId).toList();

        List<SysDataset> grandChildren = this.list(new LambdaQueryWrapper<SysDataset>()
                .in(SysDataset::getParentId, childIds)
                .select(SysDataset::getParentId));
        Set<Long> hasChildIds = grandChildren.stream()
                .map(SysDataset::getParentId)
                .collect(Collectors.toSet());

        Map<Long, DatasetStatistics> statsMap = getAllDatasetStats();

        return children.stream()
                .map(entity -> {
                    DatasetStatistics stats = statsMap.get(entity.getId());
                    DatasetVO vo = datasetConverter.entity2Vo(entity, stats);
                    vo.setHasChildren(hasChildIds.contains(entity.getId()));
                    vo.setTotal(stats != null ? stats.getFileCount() : 0L);
                    vo.setChildren(Collections.emptyList());
                    return vo;
                })
                .toList();
    }

    @Override
    public DatasetVO addDataset(DatasetAddForm dataset) {
        Long parentId = dataset.getParentId() != null ? dataset.getParentId() : SystemConstants.ROOT_NODE_ID;
        boolean exists = this.count(new LambdaQueryWrapper<SysDataset>()
                .eq(SysDataset::getName, dataset.getName())
                .eq(SysDataset::getParentId, parentId)) > 0;
        if (exists) {
            throw new BusinessException("同父节点下已存在相同名称的数据集");
        }

        SysDataset sysDataset = datasetConverter.form2Entity(dataset);
        if (this.save(sysDataset)) {
            evictAllDatasetsCache();
            DatasetStatistics stats = getAllDatasetStats().get(sysDataset.getId());
            return datasetConverter.entity2Vo(sysDataset, stats);
        } else {
            throw new BusinessException("新增数据集失败");
        }
    }

    @Override
    public DatasetVO updateDataset(Long id, DatasetUpdateForm form) {
        if (id == null || id <= 0) {
            throw new BusinessException("数据集ID无效");
        }

        SysDataset currentDataset = this.getById(id);
        if (currentDataset == null) {
            throw new BusinessException("数据集不存在");
        }

        if (form.getName() != null && !form.getName().equals(currentDataset.getName())) {
            boolean exists = this.count(new LambdaQueryWrapper<SysDataset>()
                    .eq(SysDataset::getName, form.getName())
                    .eq(SysDataset::getParentId, currentDataset.getParentId())
                    .ne(SysDataset::getId, id)) > 0;
            if (exists) {
                throw new BusinessException("同父节点下已存在相同名称的数据集");
            }
        }

        SysDataset sysDataset = datasetConverter.updateForm2Entity(form);
        sysDataset.setId(id);

        if (this.updateById(sysDataset)) {
            evictAllDatasetsCache();
            DatasetStatistics stats = getAllDatasetStats().get(id);
            return datasetConverter.entity2Vo(sysDataset, stats);
        } else {
            throw new BusinessException("更新数据集失败");
        }
    }

    @Override
    public void deleteDataset(Long id) {
        if (id == null || id <= 0) {
            throw new BusinessException("数据集ID无效");
        }

        SysDataset dataset = this.getById(id);
        if (dataset == null) {
            throw new BusinessException("数据集不存在");
        }

        if (!this.removeById(id)) {
            throw new BusinessException("删除数据集失败");
        }
        evictAllDatasetsCache();
    }

    @Override
    @Cacheable(value = "dataset:options", key = "'all'", unless = "#result == null || #result.isEmpty()")
    public List<Option<Long>> getOptions() {
        List<SysDataset> datasets = getAllDatasets();
        return buildDatasetOptions(SystemConstants.ROOT_NODE_ID, datasets);
    }

    @Override
    public List<Long> getLeafDatasetIds() {
        List<SysDataset> allDatasets = getAllDatasets();
        return TreeDataUtils.findAllLeafIds(allDatasets, SysDataset::getId, SysDataset::getParentId);
    }

    @Override
    public List<Long> getLeafDatasetId(Long id) {
        List<SysDataset> allDatasets = getAllDatasets();
        return TreeDataUtils.findLeafIdsUnder(allDatasets, id, SysDataset::getId, SysDataset::getParentId);
    }

    @Override
    public List<Long> getDatasetAndDescendantIds(Long datasetId) {
        List<SysDataset> allDatasets = getAllDatasets();
        return TreeDataUtils.findDescendantIds(allDatasets, datasetId, SysDataset::getId, SysDataset::getParentId);
    }

    @Override
    public SysDataset getRootDataset(Long id) {
        List<SysDataset> datasets = new ArrayList<>();
        SysDataset cur = this.getById(id);
        if (cur == null) {
            return null;
        }

        List<SysDataset> allDatasets = getAllDatasets();
        Map<Long, SysDataset> idToNodeMap = allDatasets.stream()
                .collect(Collectors.toMap(SysDataset::getId, d -> d));

        dfsWithCache(cur, datasets, idToNodeMap);

        if (!datasets.isEmpty()) {
            SysDataset root = datasets.get(datasets.size() - 1);
            StringBuilder fullName = new StringBuilder(root.getName());
            StringBuilder fullDescription = new StringBuilder(root.getDescription());
            for (int i = datasets.size() - 2; i >= 0; i--) {
                SysDataset dataset = datasets.get(i);
                fullName.append("/").append(dataset.getName());
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
        SysDataset cur = this.getById(id);
        if (cur == null) {
            return null;
        }

        List<SysDataset> allDatasets = getAllDatasets();
        Map<Long, SysDataset> idToNodeMap = allDatasets.stream()
                .collect(Collectors.toMap(SysDataset::getId, d -> d));

        dfsWithCache(cur, datasets, idToNodeMap);

        if (!datasets.isEmpty()) {
            StringBuilder fullName = new StringBuilder();
            StringBuilder fullDescription = new StringBuilder();
            for (int i = datasets.size() - 1; i >= 0; i--) {
                fullName.append(datasets.get(i).getName()).append("/");
                fullDescription.append(datasets.get(i).getDescription()).append("\n");
            }
            if (fullName.length() > 1) {
                fullName.setLength(fullName.length() - 1);
            }
            cur.setName(fullName.toString());
            cur.setDescription(fullDescription.toString());
        }
        return cur;
    }

    private void dfsWithCache(SysDataset cur, List<SysDataset> datasets, Map<Long, SysDataset> idToNodeMap) {
        datasets.add(cur);
        if (cur.getParentId() == null) {
            throw new BusinessException("数据集结构出现问题");
        }
        if (!cur.getParentId().equals(SystemConstants.ROOT_NODE_ID)) {
            SysDataset parent = idToNodeMap.get(cur.getParentId());
            if (parent == null) {
                throw new BusinessException("数据集结构出现问题：无法找到父节点，parentId=" + cur.getParentId());
            }
            dfsWithCache(parent, datasets, idToNodeMap);
        }
    }

    private List<Option<Long>> buildDatasetOptions(Long rootNodeId, List<SysDataset> datasets) {
        Map<Long, List<SysDataset>> parentToChildrenMap = datasets.stream()
                .collect(Collectors.groupingBy(SysDataset::getParentId));
        return buildOptionsFromMap(rootNodeId, parentToChildrenMap);
    }

    private List<Option<Long>> buildOptionsFromMap(Long parentId, Map<Long, List<SysDataset>> parentToChildrenMap) {
        List<Option<Long>> options = new ArrayList<>();
        List<SysDataset> children = parentToChildrenMap.getOrDefault(parentId, Collections.emptyList());
        for (SysDataset dataset : children) {
            Option<Long> option = new Option<>(dataset.getId(), dataset.getName());
            List<Option<Long>> subDatasets = buildOptionsFromMap(dataset.getId(), parentToChildrenMap);
            if (subDatasets != null && !subDatasets.isEmpty()) {
                option.setChildren(subDatasets);
            }
            options.add(option);
        }
        return options;
    }

    @Override
    public DatasetVO getDatasetById(Long id) {
        if (id == null || id <= 0) {
            throw new BusinessException("数据集ID无效");
        }

        SysDataset dataset = this.getById(id);
        if (dataset == null) {
            throw new BusinessException("数据集不存在");
        }

        DatasetStatistics stats = getAllDatasetStats().get(id);
        return datasetConverter.entity2Vo(dataset, stats);
    }

    @Override
    public DatasetStatistics calculateStatistics(Long datasetId) {
        return getAllDatasetStats().getOrDefault(datasetId, createEmptyStats());
    }

    @Override
    public void incrementUsageCount(Long id) {
        this.baseMapper.incrementUsageCount(id);
    }

    @Override
    public List<SysItemFile> getDatasetImages(Long datasetId, boolean recursive) {
        if (recursive) {
            List<Long> datasetIds = this.getLeafDatasetId(datasetId);
            return baseMapper.getDatasetImages(datasetIds);
        } else {
            return baseMapper.getDatasetImages(List.of(datasetId));
        }
    }

    @Override
    public String getDatasetNameByItemId(Long itemId) {
        if (itemId == null || itemId <= 0) {
            throw new BusinessException("数据项ID无效");
        }

        var datasetItem = sysDatasetItemService.getById(itemId);
        if (datasetItem == null) {
            throw new BusinessException("数据项不存在，itemId: " + itemId);
        }

        SysDataset dataset = this.getById(datasetItem.getDatasetId());
        if (dataset == null) {
            log.warn("数据项关联的数据集不存在，itemId: {}, datasetId: {}", itemId, datasetItem.getDatasetId());
            return "";
        }

        return dataset.getName();
    }

    @Override
    public void evictDatasetStatsCache(Long datasetId) {
        evictAllDatasetsCache();
    }

    @Override
    public void evictDatasetAndAncestorStatsCache(Long datasetId) {
        evictAllDatasetsCache();
    }
}
