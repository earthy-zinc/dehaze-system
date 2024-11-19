package com.pei.dehaze.service.impl;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.io.FileUtil;
import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.constant.SystemConstants;
import com.pei.dehaze.common.enums.StatusEnum;
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
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

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

        List<Long> rootIds = TreeDataUtils.findRootIds(algorithms, SysAlgorithm::getId, SysAlgorithm::getParentId);

        return rootIds.stream()
                .flatMap(rootId -> buildAlgorithmTree(rootId, algorithms).stream())
                .toList();
    }

    @Override
    public List<Option<Long>> getOption() {
        List<SysAlgorithm> algorithms = this.list(new LambdaQueryWrapper<>());
        return buildAlgorithmOptions(SystemConstants.ROOT_NODE_ID, algorithms);
    }

    @Override
    public boolean addAlgorithm(AlgorithmForm algorithm) {
        SysAlgorithm sysAlgorithm = algorithmConverter.form2Entity(algorithm);
        sysAlgorithm.setStatus(StatusEnum.ENABLE.getValue());
        if (FileUtil.isFile(sysAlgorithm.getPath())) {
            sysAlgorithm.setSize(FileUploadUtils.fileSize(sysAlgorithm.getPath()));
        }
        return this.save(sysAlgorithm);
    }

    @Override
    public boolean updateAlgorithm(AlgorithmForm algorithm) {
        SysAlgorithm sysAlgorithm = algorithmConverter.form2Entity(algorithm);
        sysAlgorithm.setSize(FileUploadUtils.fileSize(sysAlgorithm.getPath()));
        return this.updateById(sysAlgorithm);
    }

    @Override
    public boolean deleteAlgorithms(List<Long> ids) {
        List<SysAlgorithm> children = this.list(new LambdaQueryWrapper<SysAlgorithm>()
                .in(SysAlgorithm::getParentId, ids));
        List<Long> childrenIds = children.stream().map(SysAlgorithm::getId).toList();
        return this.removeByIds(CollUtil.addAll(ids, childrenIds));
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

    private List<Option<Long>> buildAlgorithmOptions(Long parentId, List<SysAlgorithm> algorithms) {
        List<Option<Long>> algorithmOptions = new ArrayList<>();
        for (SysAlgorithm algorithm : algorithms) {
            if (algorithm.getParentId().equals(parentId)) {
                Option<Long> option = new Option<>(algorithm.getId(), algorithm.getName());
                List<Option<Long>> subAlgorithms = buildAlgorithmOptions(algorithm.getId(), algorithms);
                if (CollUtil.isNotEmpty(subAlgorithms)) {
                    option.setChildren(subAlgorithms);
                }
                algorithmOptions.add(option);
            }
        }
        return algorithmOptions;
    }
}
