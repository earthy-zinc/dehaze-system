package com.pei.dehaze.service.impl;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.io.FileUtil;
import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.constant.SystemConstants;
import com.pei.dehaze.common.enums.StatusEnum;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.model.Option;
import com.pei.dehaze.common.util.FileUploadUtils;
import com.pei.dehaze.common.util.TreeDataUtils;
import com.pei.dehaze.converter.AlgorithmConverter;
import com.pei.dehaze.mapper.SysAlgorithmMapper;
import com.pei.dehaze.model.entity.SysAlgorithm;
import com.pei.dehaze.model.form.AlgorithmForm;
import com.pei.dehaze.model.query.AlgorithmQuery;
import com.pei.dehaze.model.vo.AlgorithmVO;
import com.pei.dehaze.service.SysAlgorithmService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * @author earthy-zinc
 * @since 2024-06-08 18:35:44
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class SysAlgorithmServiceImpl extends ServiceImpl<SysAlgorithmMapper, SysAlgorithm> implements SysAlgorithmService {

    private final AlgorithmConverter algorithmConverter;

    @Override
    @Cacheable(value = "algorithm:all", key = "'all'", unless = "#result == null || #result.isEmpty()")
    public List<SysAlgorithm> getAllAlgorithms() {
        return this.list(new LambdaQueryWrapper<SysAlgorithm>()
                .orderByAsc(SysAlgorithm::getId));
    }

    @CacheEvict(value = {"algorithm:all", "algorithm:list", "algorithm:options"}, allEntries = true)
    public void evictAllAlgorithmsCache() {
        log.debug("清除所有算法缓存");
    }

    @Override
    @Cacheable(value = "algorithm:list", key = "#queryParams.keywords", unless = "#result == null || #result.isEmpty()")
    public List<AlgorithmVO> getList(AlgorithmQuery queryParams) {
        List<SysAlgorithm> algorithms = this.list(new LambdaQueryWrapper<SysAlgorithm>()
                .like(CharSequenceUtil.isNotBlank(queryParams.getKeywords()), SysAlgorithm::getName, queryParams.getKeywords()));

        List<Long> rootIds = TreeDataUtils.findRootIds(algorithms, SysAlgorithm::getId, SysAlgorithm::getParentId);

        // 构建 parentId -> children Map，避免 O(n²) 递归
        Map<Long, List<SysAlgorithm>> parentToChildrenMap = algorithms.stream()
                .collect(Collectors.groupingBy(SysAlgorithm::getParentId));

        return rootIds.stream()
                .flatMap(rootId -> buildAlgorithmTree(rootId, parentToChildrenMap).stream())
                .toList();
    }

    /**
     * 根据 parentId，获取其根节点对应的SysAlgorithm
     * 使用缓存的 getAllAlgorithms 构建 Map，避免 N+1 查询
     * @return 根节点对应的SysAlgorithm
     */
    @Override
    public SysAlgorithm getRootAlgorithm(Long id) {
        SysAlgorithm algorithm = this.getById(id);
        if (algorithm == null) {
            throw new BusinessException("当前算法不存在");
        }

        // 使用缓存的全部算法列表构建 Map，避免逐级 getById 查询
        Map<Long, SysAlgorithm> idToNodeMap = getAllAlgorithms().stream()
                .collect(Collectors.toMap(SysAlgorithm::getId, a -> a));

        while (!algorithm.getParentId().equals(SystemConstants.ROOT_NODE_ID)) {
            SysAlgorithm parent = idToNodeMap.get(algorithm.getParentId());
            if (parent == null) {
                throw new BusinessException("无法获取算法根节点");
            }
            algorithm = parent;
        }
        return algorithm;
    }

    @Override
    @Cacheable(value = "algorithm:options", key = "'all'", unless = "#result == null || #result.isEmpty()")
    public List<Option<Long>> getOption() {
        List<SysAlgorithm> algorithms = this.list(
                new LambdaQueryWrapper<SysAlgorithm>()
                        .eq(SysAlgorithm::getStatus, StatusEnum.ENABLE.getValue()));
        // 使用 Map 分组，避免 O(n²) 递归
        Map<Long, List<SysAlgorithm>> parentToChildrenMap = algorithms.stream()
                .collect(Collectors.groupingBy(SysAlgorithm::getParentId));
        return buildAlgorithmOptions(SystemConstants.ROOT_NODE_ID, parentToChildrenMap);
    }

    @Override
    public SysAlgorithm getAlgorithmById(Long id) {
        SysAlgorithm cur = this.getById(id);
        if (cur == null) {
            throw new BusinessException("当前算法不存在");
        }

        // 使用缓存的全部算法列表构建 Map，避免逐级 getById 查询
        Map<Long, SysAlgorithm> idToNodeMap = getAllAlgorithms().stream()
                .collect(Collectors.toMap(SysAlgorithm::getId, a -> a));

        List<SysAlgorithm> algorithms = new ArrayList<>();
        collectAncestors(cur, algorithms, idToNodeMap);

        if (!algorithms.isEmpty()) {
            StringBuilder fullName = new StringBuilder();
            StringBuilder fullDescription = new StringBuilder();
            for (int i = algorithms.size() - 1; i >= 0; i--) { // 从根节点到当前节点
                fullName.append(algorithms.get(i).getName()).append("/");
                fullDescription.append(algorithms.get(i).getDescription()).append("\n");
            }
            if (fullName.length() > 4) {
                fullName.setLength(fullName.length() - 1);
            }
            cur.setName(fullName.toString());
            cur.setDescription(fullDescription.toString());
        }
        return cur;
    }

    @Override
    @CacheEvict(value = {"algorithm:all", "algorithm:list", "algorithm:options"}, allEntries = true)
    public boolean addAlgorithm(AlgorithmForm algorithm) {
        SysAlgorithm sysAlgorithm = algorithmConverter.form2Entity(algorithm);
        sysAlgorithm.setStatus(StatusEnum.ENABLE.getValue());
        if (FileUtil.isFile(sysAlgorithm.getPath())) {
            sysAlgorithm.setSize(FileUploadUtils.fileSize(sysAlgorithm.getPath()));
        }
        return this.save(sysAlgorithm);
    }

    private void collectAncestors(SysAlgorithm cur, List<SysAlgorithm> algorithms,
                                  Map<Long, SysAlgorithm> idToNodeMap) {
        algorithms.add(cur);
        if (cur.getParentId() == null) {
            throw new BusinessException("算法结构出现问题");
        }
        if (!cur.getParentId().equals(SystemConstants.ROOT_NODE_ID)) {
            SysAlgorithm parent = idToNodeMap.get(cur.getParentId());
            if (parent == null) {
                throw new BusinessException("算法结构出现问题：无法找到父节点，parentId=" + cur.getParentId());
            }
            collectAncestors(parent, algorithms, idToNodeMap);
        }
    }

    @Override
    @CacheEvict(value = {"algorithm:all", "algorithm:list", "algorithm:options"}, allEntries = true)
    public boolean updateAlgorithm(AlgorithmForm algorithm) {
        SysAlgorithm sysAlgorithm = algorithmConverter.form2Entity(algorithm);
        sysAlgorithm.setSize(FileUploadUtils.fileSize(sysAlgorithm.getPath()));
        return this.updateById(sysAlgorithm);
    }

    @Override
    @CacheEvict(value = {"algorithm:all", "algorithm:list", "algorithm:options"}, allEntries = true)
    public boolean deleteAlgorithms(List<Long> ids) {
        List<SysAlgorithm> children = this.list(new LambdaQueryWrapper<SysAlgorithm>()
                .in(SysAlgorithm::getParentId, ids));
        List<Long> childrenIds = children.stream().map(SysAlgorithm::getId).toList();
        return this.removeByIds(CollUtil.addAll(ids, childrenIds));
    }

    /**
     * 使用 Map 分组构建算法树，时间复杂度 O(n)
     */
    private List<AlgorithmVO> buildAlgorithmTree(Long parentId, Map<Long, List<SysAlgorithm>> parentToChildrenMap) {
        List<SysAlgorithm> children = parentToChildrenMap.getOrDefault(parentId, Collections.emptyList());
        return children.stream()
                .map(entity -> {
                    AlgorithmVO algorithmVO = algorithmConverter.entity2Vo(entity);
                    algorithmVO.setChildren(buildAlgorithmTree(entity.getId(), parentToChildrenMap));
                    return algorithmVO;
                }).toList();
    }

    /**
     * 使用 Map 分组构建算法下拉选项，时间复杂度 O(n)
     */
    private List<Option<Long>> buildAlgorithmOptions(Long parentId, Map<Long, List<SysAlgorithm>> parentToChildrenMap) {
        List<SysAlgorithm> children = parentToChildrenMap.getOrDefault(parentId, Collections.emptyList());
        List<Option<Long>> algorithmOptions = new ArrayList<>();
        for (SysAlgorithm algorithm : children) {
            Option<Long> option = new Option<>(algorithm.getId(), algorithm.getName());
            List<Option<Long>> subAlgorithms = buildAlgorithmOptions(algorithm.getId(), parentToChildrenMap);
            if (CollUtil.isNotEmpty(subAlgorithms)) {
                option.setChildren(subAlgorithms);
            }
            algorithmOptions.add(option);
        }
        return algorithmOptions;
    }
}
