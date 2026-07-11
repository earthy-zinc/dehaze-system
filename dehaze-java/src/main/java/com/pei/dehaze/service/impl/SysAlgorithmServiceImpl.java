package com.pei.dehaze.service.impl;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.io.FileUtil;
import cn.hutool.core.text.CharSequenceUtil;
import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.constant.JwtClaimConstants;
import com.pei.dehaze.common.constant.SystemConstants;
import com.pei.dehaze.common.enums.AlgorithmStatusEnum;
import com.pei.dehaze.common.enums.StatusEnum;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.model.Option;
import com.pei.dehaze.common.util.FileUploadUtils;
import com.pei.dehaze.common.util.TreeDataUtils;
import com.pei.dehaze.converter.AlgorithmConverter;
import com.pei.dehaze.mapper.SysAlgorithmMapper;
import com.pei.dehaze.mapper.SysEvalLogMapper;
import com.pei.dehaze.mapper.SysPredLogMapper;
import com.pei.dehaze.model.entity.SysAlgorithm;
import com.pei.dehaze.model.entity.SysEvalLog;
import com.pei.dehaze.model.entity.SysPredLog;
import com.pei.dehaze.model.form.AlgorithmAuditForm;
import com.pei.dehaze.model.form.AlgorithmForm;
import com.pei.dehaze.model.query.AlgorithmQuery;
import com.pei.dehaze.model.vo.AlgorithmMonitorVO;
import com.pei.dehaze.model.vo.AlgorithmVO;
import com.pei.dehaze.service.SysAlgorithmService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.*;
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
    private final SysPredLogMapper sysPredLogMapper;
    private final SysEvalLogMapper sysEvalLogMapper;

    @Override
    public List<SysAlgorithm> getAllAlgorithms() {
        return this.list(new LambdaQueryWrapper<SysAlgorithm>()
                .orderByAsc(SysAlgorithm::getId));
    }

    @Override
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
    public Long addAlgorithm(AlgorithmForm algorithm) {
        SysAlgorithm sysAlgorithm = algorithmConverter.form2Entity(algorithm);
        sysAlgorithm.setStatus(StatusEnum.ENABLE.getValue());
        if (FileUtil.isFile(sysAlgorithm.getPath())) {
            sysAlgorithm.setSize(FileUploadUtils.fileSize(sysAlgorithm.getPath()));
        }
        this.save(sysAlgorithm);
        return sysAlgorithm.getId();
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

    // ==================== 状态管理 ====================

    @Override
    public boolean updateStatus(Long id, Integer status) {
        SysAlgorithm algorithm = this.getById(id);
        if (algorithm == null) {
            throw new BusinessException("算法不存在");
        }

        AlgorithmStatusEnum targetStatus = Arrays.stream(AlgorithmStatusEnum.values())
                .filter(e -> e.getValue().equals(status))
                .findFirst()
                .orElse(null);
        if (targetStatus == null) {
            throw new BusinessException("无效的状态值: " + status);
        }

        // 状态流转校验
        validateStatusTransition(algorithm.getStatus(), status);

        algorithm.setStatus(status);
        return this.updateById(algorithm);
    }

    @Override
    public boolean auditAlgorithm(Long id, AlgorithmAuditForm form) {
        SysAlgorithm algorithm = this.getById(id);
        if (algorithm == null) {
            throw new BusinessException("算法不存在");
        }

        if (!AlgorithmStatusEnum.PENDING_REVIEW.getValue().equals(algorithm.getStatus())) {
            throw new BusinessException("只有待审核状态的算法才能进行审核操作");
        }

        if (Boolean.TRUE.equals(form.getApproved())) {
            // 审核通过 → 已发布
            algorithm.setStatus(AlgorithmStatusEnum.PUBLISHED.getValue());
        } else {
            // 审核驳回 → 测试中
            if (CharSequenceUtil.isBlank(form.getRemark())) {
                throw new BusinessException("驳回时必须填写审核备注");
            }
            algorithm.setStatus(AlgorithmStatusEnum.TESTING.getValue());
            algorithm.setAuditRemark(form.getRemark());
        }

        algorithm.setAuditBy(getCurrentUserId());
        algorithm.setAuditTime(LocalDateTime.now());

        return this.updateById(algorithm);
    }

    @Override
    public AlgorithmMonitorVO getMonitorData(Long id) {
        SysAlgorithm algorithm = this.getById(id);
        if (algorithm == null) {
            throw new BusinessException("算法不存在");
        }

        AlgorithmMonitorVO monitor = new AlgorithmMonitorVO();

        // 统计预测日志
        long totalCalls = sysPredLogMapper.selectCount(new LambdaQueryWrapper<SysPredLog>()
                .eq(SysPredLog::getAlgorithmId, id));
        monitor.setCallCount(totalCalls);

        // 今日调用次数
        long todayCalls = sysPredLogMapper.selectCount(new LambdaQueryWrapper<SysPredLog>()
                .eq(SysPredLog::getAlgorithmId, id)
                .ge(SysPredLog::getCreateTime, LocalDateTime.now().withHour(0).withMinute(0).withSecond(0)));
        monitor.setTodayCallCount(todayCalls);

        // 平均处理时间
        List<SysPredLog> predLogs = sysPredLogMapper.selectList(new LambdaQueryWrapper<SysPredLog>()
                .eq(SysPredLog::getAlgorithmId, id)
                .isNotNull(SysPredLog::getTime));
        double avgTime = predLogs.stream()
                .mapToInt(SysPredLog::getTime)
                .average()
                .orElse(0.0);
        monitor.setAvgTime(Math.round(avgTime * 100.0) / 100.0);

        // 成功率：有predUrl的日志视为成功
        long successCount = predLogs.stream()
                .filter(p -> CharSequenceUtil.isNotBlank(p.getPredUrl()))
                .count();
        double successRate = totalCalls > 0 ? (double) successCount / totalCalls * 100 : 100.0;
        monitor.setSuccessRate(Math.round(successRate * 100.0) / 100.0);

        return monitor;
    }

    @Override
    public String exportAlgorithmJson(Long id) {
        SysAlgorithm algorithm = this.getById(id);
        if (algorithm == null) {
            throw new BusinessException("算法不存在");
        }

        // 获取父算法名称用于导入参考
        String parentName = "";
        if (algorithm.getParentId() != null && algorithm.getParentId() > 0) {
            SysAlgorithm parent = this.getById(algorithm.getParentId());
            parentName = parent != null ? parent.getName() : "";
        }

        Map<String, Object> exportData = new LinkedHashMap<>();
        exportData.put("formatVersion", "1.0");
        exportData.put("name", algorithm.getName());
        exportData.put("type", algorithm.getType());
        exportData.put("parentName", parentName);
        exportData.put("version", algorithm.getVersion());
        exportData.put("description", algorithm.getDescription());
        exportData.put("importPath", algorithm.getImportPath());
        exportData.put("flops", algorithm.getFlops());
        exportData.put("params", algorithm.getParams());
        exportData.put("status", algorithm.getStatus());
        exportData.put("statusLabel", Arrays.stream(AlgorithmStatusEnum.values())
                .filter(e -> e.getValue().equals(algorithm.getStatus()))
                .map(AlgorithmStatusEnum::getLabel)
                .findFirst().orElse(""));
        exportData.put("exportTime", LocalDateTime.now().toString());

        return JSONUtil.toJsonPrettyStr(exportData);
    }

    /**
     * 从 SecurityContext 获取当前登录用户 ID
     */
    private Long getCurrentUserId() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth != null && auth.getPrincipal() instanceof com.pei.dehaze.security.model.SysUserDetails userDetails) {
            return userDetails.getUserId();
        }
        // 尝试从 JWT claims 获取
        if (auth != null && auth.getDetails() instanceof Map<?, ?> details) {
            Object userId = details.get(JwtClaimConstants.USER_ID);
            if (userId instanceof Long) return (Long) userId;
            if (userId instanceof Number) return ((Number) userId).longValue();
        }
        log.warn("无法获取当前用户ID，使用默认值");
        return 0L;
    }

    /**
     * 校验状态流转合法性
     */
    private void validateStatusTransition(Integer currentStatus, Integer targetStatus) {
        if (currentStatus == null) {
            return;
        }
        // 终态不允许变更
        if (AlgorithmStatusEnum.FINAL_STATUSES.contains(currentStatus)
                && !AlgorithmStatusEnum.ARCHIVED.getValue().equals(currentStatus)) {
            throw new BusinessException("终态算法不允许修改状态");
        }
        // 不允许直接跳转到已发布
        if (AlgorithmStatusEnum.PUBLISHED.getValue().equals(targetStatus)
                && !AlgorithmStatusEnum.PENDING_REVIEW.getValue().equals(currentStatus)) {
            throw new BusinessException("算法必须经过审核才能发布");
        }
    }
}
