package com.pei.dehaze.service.impl;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.constant.SystemConstants;
import com.pei.dehaze.common.model.Option;
import com.pei.dehaze.converter.AlgorithmConverter;
import com.pei.dehaze.mapper.SysAlgorithmMapper;
import com.pei.dehaze.model.entity.SysAlgorithm;
import com.pei.dehaze.model.query.AlgorithmQuery;
import com.pei.dehaze.model.vo.AlgorithmVO;
import com.pei.dehaze.service.SysAlgorithmService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * @author earthy-zinc
 * @since 2024-06-08 18:35:44
 */
@Service
@RequiredArgsConstructor
public class SysAlgorithmServiceImpl extends ServiceImpl<SysAlgorithmMapper, SysAlgorithm> implements SysAlgorithmService {

    private final AlgorithmConverter algorithmConverter;

    @Override
    public List<AlgorithmVO> getList(AlgorithmQuery queryParams) {
        List<SysAlgorithm> algorithms = this.list(new LambdaQueryWrapper<SysAlgorithm>()
                .like(CharSequenceUtil.isNotBlank(queryParams.getKeywords()), SysAlgorithm::getName, queryParams.getKeywords()));

        Set<Long> algorithmIds = algorithms.stream()
                .map(SysAlgorithm::getId)
                .collect(Collectors.toSet());

        Set<Long> parentIds = algorithms.stream()
                .map(SysAlgorithm::getParentId)
                .collect(Collectors.toSet());

        List<Long> rootIds = parentIds.stream()
                .filter(id -> !algorithmIds.contains(id))
                .toList();

        return rootIds.stream()
                .flatMap(rootId -> buildAlgorithmTree(rootId, algorithms).stream())
                .toList();
    }

    @Override
    public List<Option> getOption() {
        List<SysAlgorithm> algorithms = this.list(new LambdaQueryWrapper<>());
        return buildAlgorithmOptions(SystemConstants.ROOT_NODE_ID, algorithms);
    }

    private List<AlgorithmVO> buildAlgorithmTree(Long rootId, List<SysAlgorithm> algorithms) {
        return CollUtil.emptyIfNull(algorithms)
                .stream()
                .filter(algorithm -> algorithm.getParentId().equals(rootId))
                .map(entity -> {
                    AlgorithmVO algorithmVO = algorithmConverter.entity2Vo(entity);
                    algorithmVO.setChildren(buildAlgorithmTree(entity.getId(), algorithms));
                    return algorithmVO;
                }).toList();
    }

    private List<Option> buildAlgorithmOptions(Long parentId, List<SysAlgorithm> algorithms) {
        List<Option> algorithmOptions = new ArrayList<>();
        for (SysAlgorithm algorithm : algorithms) {
            if (algorithm.getParentId().equals(parentId)) {
                Option option = new Option<>(algorithm.getId(), algorithm.getName());
                List<Option> subAlgorithms = buildAlgorithmOptions(algorithm.getId(), algorithms);
                if (CollUtil.isNotEmpty(subAlgorithms)) {
                    option.setChildren(subAlgorithms);
                }
                algorithmOptions.add(option);
            }
        }
        return algorithmOptions;
    }
}
