package com.pei.dehaze.service.impl;

import cn.hutool.core.util.StrUtil;
import cn.hutool.extra.pinyin.PinyinUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.enums.AlgorithmStatusEnum;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysPredLogMapper;
import com.pei.dehaze.mapper.SysRatingMapper;
import com.pei.dehaze.model.entity.SysAlgorithm;
import com.pei.dehaze.model.entity.SysPredLog;
import com.pei.dehaze.model.entity.SysRating;
import com.pei.dehaze.model.form.AlgorithmCompareForm;
import com.pei.dehaze.model.form.AlgorithmTestForm;
import com.pei.dehaze.model.form.PredictionForm;
import com.pei.dehaze.model.vo.AlgorithmCompareVO;
import com.pei.dehaze.model.vo.AlgorithmDetailVO;
import com.pei.dehaze.model.vo.AlgorithmSelectNodeVO;
import com.pei.dehaze.model.vo.PredictionResultVO;
import com.pei.dehaze.service.AlgorithmSelectService;
import com.pei.dehaze.service.SysAlgorithmService;
import com.pei.dehaze.service.SysPredLogService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class AlgorithmSelectServiceImpl implements AlgorithmSelectService {

    private final SysAlgorithmService sysAlgorithmService;
    private final SysPredLogService sysPredLogService;
    private final SysPredLogMapper sysPredLogMapper;
    private final SysRatingMapper sysRatingMapper;

    @Override
    public List<AlgorithmSelectNodeVO> getTree() {
        List<SysAlgorithm> published = sysAlgorithmService.getAllAlgorithms().stream()
                .filter(a -> AlgorithmStatusEnum.PUBLISHED.getValue().equals(a.getStatus()))
                .toList();

        if (published.isEmpty()) {
            return Collections.emptyList();
        }

        Set<Long> publishedIds = published.stream()
                .map(SysAlgorithm::getId)
                .collect(Collectors.toSet());

        Map<Long, List<SysAlgorithm>> parentToChildrenMap = published.stream()
                .collect(Collectors.groupingBy(SysAlgorithm::getParentId));

        return buildTree(0L, parentToChildrenMap, publishedIds);
    }

    private List<AlgorithmSelectNodeVO> buildTree(Long parentId,
                                                   Map<Long, List<SysAlgorithm>> parentToChildrenMap,
                                                   Set<Long> publishedIds) {
        List<SysAlgorithm> children = parentToChildrenMap.getOrDefault(parentId, Collections.emptyList());
        List<AlgorithmSelectNodeVO> nodes = new ArrayList<>();
        for (SysAlgorithm alg : children) {
            AlgorithmSelectNodeVO node = new AlgorithmSelectNodeVO();
            node.setId(alg.getId());
            node.setParentId(alg.getParentId());
            node.setName(alg.getName());
            node.setType(alg.getType());
            List<AlgorithmSelectNodeVO> subNodes = buildTree(alg.getId(), parentToChildrenMap, publishedIds);
            // 过滤空分类节点：无子节点的分类节点隐藏（分类节点 parentId=0 且有子节点）
            boolean isCategory = alg.getParentId() != null && alg.getParentId() == 0;
            boolean isLeaf = subNodes.isEmpty();
            if (isCategory && isLeaf) {
                continue;
            }
            node.setLeaf(isLeaf);
            if (!isLeaf) {
                node.setChildren(subNodes);
            }
            nodes.add(node);
        }
        return nodes;
    }

    @Override
    public AlgorithmDetailVO getDetail(Long id) {
        SysAlgorithm algorithm = sysAlgorithmService.getAlgorithmById(id);
        if (!AlgorithmStatusEnum.PUBLISHED.getValue().equals(algorithm.getStatus())) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "算法不存在或未发布");
        }

        AlgorithmDetailVO vo = new AlgorithmDetailVO();
        vo.setId(algorithm.getId());
        vo.setName(algorithm.getName());
        vo.setType(algorithm.getType());
        vo.setImg(algorithm.getImg());
        vo.setDescription(algorithm.getDescription());
        vo.setPath(algorithm.getPath());
        vo.setSize(algorithm.getSize());
        vo.setParams(algorithm.getParams());
        vo.setFlops(algorithm.getFlops());
        vo.setVersion(algorithm.getVersion());
        vo.setStatus(algorithm.getStatus());

        // 评分聚合
        List<SysRating> ratings = sysRatingMapper.selectList(
                new LambdaQueryWrapper<SysRating>()
                        .eq(SysRating::getAlgorithmId, id));
        long ratingCount = ratings.size();
        double avgRating = ratings.stream()
                .filter(r -> r.getRating() != null)
                .mapToInt(SysRating::getRating)
                .average()
                .orElse(0.0);
        vo.setAvgRating(Math.round(avgRating * 10.0) / 10.0);
        vo.setRatingCount(ratingCount);

        // 使用次数
        long usageCount = sysPredLogMapper.selectCount(
                new LambdaQueryWrapper<SysPredLog>()
                        .eq(SysPredLog::getAlgorithmId, id));
        vo.setUsageCount(usageCount);

        // 样例效果图：从 sys_pred_log 取最近3条成功记录的结果URL
        List<String> sampleImages = sysPredLogMapper.selectList(
                new LambdaQueryWrapper<SysPredLog>()
                        .eq(SysPredLog::getAlgorithmId, id)
                        .isNotNull(SysPredLog::getPredUrl)
                        .ne(SysPredLog::getPredUrl, "")
                        .orderByDesc(SysPredLog::getCreateTime)
                        .last("LIMIT 3"))
                .stream()
                .map(SysPredLog::getPredUrl)
                .filter(StrUtil::isNotBlank)
                .toList();
        vo.setSampleImages(sampleImages);

        return vo;
    }

    @Override
    public PredictionResultVO test(Long algorithmId, AlgorithmTestForm form) {
        SysAlgorithm algorithm = sysAlgorithmService.getAlgorithmById(algorithmId);
        if (!AlgorithmStatusEnum.PUBLISHED.getValue().equals(algorithm.getStatus())) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "算法不存在或未发布");
        }

        PredictionForm predForm = new PredictionForm();
        predForm.setAlgorithmId(algorithmId);
        predForm.setFileId(form.getFileId());
        predForm.setImageUrl(form.getImageUrl());
        predForm.setParams(form.getParams());

        return sysPredLogService.predict(predForm);
    }

    @Override
    public List<AlgorithmSelectNodeVO> search(String keyword) {
        if (StrUtil.isBlank(keyword)) {
            return Collections.emptyList();
        }

        List<SysAlgorithm> published = sysAlgorithmService.getAllAlgorithms().stream()
                .filter(a -> AlgorithmStatusEnum.PUBLISHED.getValue().equals(a.getStatus()))
                .toList();

        if (published.isEmpty()) {
            return Collections.emptyList();
        }

        String kw = keyword.trim().toLowerCase();
        // 拼音首字母
        String pinyinFirst = PinyinUtil.getFirstLetter(kw, "");

        List<SysAlgorithm> matched = published.stream()
                .filter(a -> matchesKeyword(a, kw, pinyinFirst))
                .toList();

        return matched.stream().map(a -> {
            AlgorithmSelectNodeVO node = new AlgorithmSelectNodeVO();
            node.setId(a.getId());
            node.setParentId(a.getParentId());
            node.setName(a.getName());
            node.setType(a.getType());
            node.setLeaf(true);
            return node;
        }).toList();
    }

    private boolean matchesKeyword(SysAlgorithm a, String kw, String pinyinFirst) {
        if (StrUtil.isBlank(a.getName())) {
            return false;
        }
        String name = a.getName().toLowerCase();
        if (name.contains(kw)) {
            return true;
        }
        // 拼音全拼匹配
        String fullPinyin = PinyinUtil.getPinyin(name, "");
        if (fullPinyin != null && fullPinyin.contains(kw)) {
            return true;
        }
        // 拼音首字母匹配
        if (StrUtil.isNotBlank(pinyinFirst)) {
            String nameFirst = PinyinUtil.getFirstLetter(name, "");
            if (nameFirst != null && nameFirst.contains(pinyinFirst)) {
                return true;
            }
        }
        // 标签匹配
        if (StrUtil.isNotBlank(a.getType())) {
            String type = a.getType().toLowerCase();
            if (type.contains(kw)) {
                return true;
            }
        }
        // 描述匹配
        if (StrUtil.isNotBlank(a.getDescription())) {
            String desc = a.getDescription().toLowerCase();
            if (desc.contains(kw)) {
                return true;
            }
        }
        return false;
    }

    @Override
    public List<AlgorithmCompareVO> compare(AlgorithmCompareForm form) {
        List<Long> algorithmIds = form.getAlgorithmIds();
        if (algorithmIds == null || algorithmIds.size() < 2) {
            throw new BusinessException(ResultCode.BUSINESS_ERROR, "对比算法数量需在2-3个之间");
        }
        if (algorithmIds.size() > 3) {
            throw new BusinessException(ResultCode.BUSINESS_ERROR, "对比算法数量不能超过3个");
        }

        // 校验算法均存在且已发布
        for (Long id : algorithmIds) {
            SysAlgorithm alg = sysAlgorithmService.getAlgorithmById(id);
            if (!AlgorithmStatusEnum.PUBLISHED.getValue().equals(alg.getStatus())) {
                throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "算法[" + alg.getName() + "]不存在或未发布");
            }
        }

        // 分别执行预测
        List<AlgorithmCompareVO> results = new ArrayList<>();
        for (Long algorithmId : algorithmIds) {
            SysAlgorithm alg = sysAlgorithmService.getAlgorithmById(algorithmId);
            AlgorithmCompareVO vo = new AlgorithmCompareVO();
            vo.setAlgorithmId(algorithmId);
            vo.setAlgorithmName(alg.getName());

            try {
                PredictionForm predForm = new PredictionForm();
                predForm.setAlgorithmId(algorithmId);
                predForm.setFileId(form.getFileId());
                predForm.setImageUrl(form.getImageUrl());

                PredictionResultVO predResult = sysPredLogService.predict(predForm);
                vo.setResultUrl(predResult.getResultUrl());
                vo.setTime(predResult.getTime());
            } catch (Exception e) {
                log.error("算法[{}]对比预测失败: {}", algorithmId, e.getMessage());
                vo.setResultUrl(null);
                vo.setTime(null);
            }
            results.add(vo);
        }

        return results;
    }
}
