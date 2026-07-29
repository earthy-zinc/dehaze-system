package com.pei.dehaze.service.impl;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.text.CharSequenceUtil;
import cn.hutool.http.HttpRequest;
import cn.hutool.http.HttpResponse;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.constant.SystemConstants;
import com.pei.dehaze.common.enums.AlgorithmStatusEnum;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.model.Option;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.common.util.TreeDataUtils;
import com.pei.dehaze.config.property.ModelProperties;
import com.pei.dehaze.converter.AlgorithmConverter;
import com.pei.dehaze.mapper.SysAlgorithmMapper;
import com.pei.dehaze.mapper.SysEvalLogMapper;
import com.pei.dehaze.mapper.SysPredLogMapper;
import com.pei.dehaze.model.entity.SysAlgorithm;
import com.pei.dehaze.model.entity.SysPredLog;
import com.pei.dehaze.model.form.AlgorithmAuditForm;
import com.pei.dehaze.model.form.AlgorithmForm;
import com.pei.dehaze.model.query.AlgorithmQuery;
import com.pei.dehaze.model.vo.AlgorithmMonitorVO;
import com.pei.dehaze.model.vo.AlgorithmVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.SysAlgorithmService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

/**
 * @author earthy-zinc
 * @since 2024-06-08 18:35:44
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class SysAlgorithmServiceImpl extends ServiceImpl<SysAlgorithmMapper, SysAlgorithm> implements SysAlgorithmService {

    private final AlgorithmConverter algorithmConverter;
    private final SysPredLogMapper sysPredLogMapper;
    private final SysEvalLogMapper sysEvalLogMapper;
    private final ModelProperties modelProperties;

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
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "当前算法不存在");
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
                        .eq(SysAlgorithm::getStatus, AlgorithmStatusEnum.PUBLISHED.getValue()));
        // 使用 Map 分组，避免 O(n²) 递归
        Map<Long, List<SysAlgorithm>> parentToChildrenMap = algorithms.stream()
                .collect(Collectors.groupingBy(SysAlgorithm::getParentId));
        return buildAlgorithmOptions(SystemConstants.ROOT_NODE_ID, parentToChildrenMap);
    }

    @Override
    public List<AlgorithmVO> listAll() {
        List<SysAlgorithm> algorithms = getAllAlgorithms();
        return algorithms.stream()
                .map(algorithmConverter::entity2Vo)
                .collect(Collectors.toList());
    }

    @Override
    public SysAlgorithm getAlgorithmById(Long id) {
        SysAlgorithm cur = this.getById(id);
        if (cur == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "算法不存在");
        }
        return cur;
    }

    @Override
    public Long addAlgorithm(AlgorithmForm algorithm) {
        SysAlgorithm sysAlgorithm = algorithmConverter.form2Entity(algorithm);
        Long sizeBytes = checkModelExists(sysAlgorithm.getPath());
        if (sizeBytes != null) {
            sysAlgorithm.setSize(readableSize(sizeBytes));
        }
        this.save(sysAlgorithm);
        return sysAlgorithm.getId();
    }

    @Override
    public boolean updateAlgorithm(AlgorithmForm algorithm) {
        SysAlgorithm sysAlgorithm = algorithmConverter.form2Entity(algorithm);
        Long sizeBytes = checkModelExists(sysAlgorithm.getPath());
        if (sizeBytes != null) {
            sysAlgorithm.setSize(readableSize(sizeBytes));
        }
        return this.updateById(sysAlgorithm);
    }

    /**
     * 通过 HTTP HEAD 请求校验模型权重文件是否可访问
     *
     * 复用 nginx-dataset 静态服务（/models 路径），sys_algorithm.path 为相对路径，
     * 拼接为 {modelProperties.baseUrl}/{path} 后发起 HEAD 请求。
     *
     * @param relativePath 算法 path 字段（如 "AECR-Net/NH_train.pk"）
     * @return 文件字节数；不可访问或异常时返回 null
     */
    private Long checkModelExists(String relativePath) {
        if (CharSequenceUtil.isBlank(relativePath)) {
            return null;
        }
        String url = modelProperties.getBaseUrl().replaceAll("/+$", "")
                + "/" + relativePath.replaceAll("^/+", "");
        try (HttpResponse response = HttpRequest.head(url)
                .timeout(modelProperties.getReadTimeout())
                .execute()) {
            if (response.getStatus() != 200) {
                return null;
            }
            String contentLength = response.header("Content-Length");
            if (CharSequenceUtil.isBlank(contentLength)) {
                return null;
            }
            return Long.parseLong(contentLength);
        } catch (Exception e) {
            log.warn("模型权重校验失败: url={}, error={}", url, e.getMessage());
            return null;
        }
    }

    /**
     * 字节数转可读大小字符串（与 Python convert_size 对齐）
     */
    private String readableSize(long bytes) {
        if (bytes == 0) {
            return "0B";
        }
        String[] units = {"B", "KB", "MB", "GB", "TB", "PB"};
        int digit = (int) (Math.log(bytes) / Math.log(1024));
        double size = bytes / Math.pow(1024, digit);
        return String.format("%.2f %s", size, units[digit]);
    }

    @Override
    public boolean deleteAlgorithms(List<Long> ids) {
        if (ids == null || ids.isEmpty()) {
            return true;
        }
        List<SysAlgorithm> allAlgorithms = this.list();
        Map<Long, SysAlgorithm> idToNodeMap = allAlgorithms.stream()
                .collect(Collectors.toMap(SysAlgorithm::getId, a -> a));
        for (Long id : ids) {
            if (!idToNodeMap.containsKey(id)) {
                throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "算法不存在");
            }
        }
        Set<Long> allIds = new HashSet<>();
        for (Long id : ids) {
            allIds.addAll(TreeDataUtils.findDescendantIds(allAlgorithms, id, SysAlgorithm::getId, SysAlgorithm::getParentId));
        }
        // 校验待删除算法的状态：只有草稿/已停用/已归档状态可删除
        for (Long id : allIds) {
            SysAlgorithm algo = idToNodeMap.get(id);
            if (algo != null && !AlgorithmStatusEnum.DELETABLE_STATUSES.contains(algo.getStatus())) {
                throw new BusinessException(ResultCode.DATA_STATE_NOT_ALLOW,
                        "算法[" + algo.getName() + "]当前状态不允许删除，请先停用或归档");
            }
        }
        return this.removeByIds(allIds);
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
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "算法不存在");
        }

        AlgorithmStatusEnum targetStatus = Arrays.stream(AlgorithmStatusEnum.values())
                .filter(e -> e.getValue().equals(status))
                .findFirst()
                .orElse(null);
        if (targetStatus == null) {
            throw new BusinessException(ResultCode.DATA_STATE_NOT_ALLOW, "无效的状态值: " + status);
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
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "算法不存在");
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
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "算法不存在");
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
    public List<Map<String, Object>> getMonitorStats(Long id, Integer days) {
        SysAlgorithm algorithm = this.getById(id);
        if (algorithm == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "算法不存在");
        }
        int rangeDays = days != null && days > 0 ? days : 7;
        LocalDateTime startTime = LocalDate.now().minusDays(rangeDays - 1).atStartOfDay();

        List<SysPredLog> logs = sysPredLogMapper.selectList(new LambdaQueryWrapper<SysPredLog>()
                .eq(SysPredLog::getAlgorithmId, id)
                .ge(SysPredLog::getCreateTime, startTime));

        // 按日期分组聚合
        Map<LocalDate, List<SysPredLog>> byDate = logs.stream()
                .collect(Collectors.groupingBy(log -> log.getCreateTime().toLocalDate()));

        List<Map<String, Object>> result = new ArrayList<>();
        for (int i = rangeDays - 1; i >= 0; i--) {
            LocalDate date = LocalDate.now().minusDays(i);
            List<SysPredLog> dayLogs = byDate.getOrDefault(date, Collections.emptyList());
            long count = dayLogs.size();
            double avgTime = dayLogs.stream()
                    .filter(l -> l.getTime() != null)
                    .mapToInt(SysPredLog::getTime)
                    .average()
                    .orElse(0.0);
            long successCount = dayLogs.stream()
                    .filter(l -> CharSequenceUtil.isNotBlank(l.getPredUrl()))
                    .count();
            double successRate = count > 0 ? (double) successCount / count * 100 : 0.0;

            Map<String, Object> dayStat = new LinkedHashMap<>();
            dayStat.put("date", date.toString());
            dayStat.put("callCount", count);
            dayStat.put("avgTime", Math.round(avgTime * 100.0) / 100.0);
            dayStat.put("successRate", Math.round(successRate * 100.0) / 100.0);
            result.add(dayStat);
        }
        return result;
    }

    /**
     * 从 SecurityContext 获取当前登录用户 ID
     */
    private Long getCurrentUserId() {
        Long userId = SecurityUtils.getUserId();
        if (userId == null) {
            throw new BusinessException(ResultCode.TOKEN_INVALID, "无法获取当前登录用户信息");
        }
        return userId;
    }

    /**
     * 校验状态流转合法性（三端统一状态机）
     * 草稿(1)→测试中(2)；测试中(2)→待审核(3)/草稿(1)；待审核(3)→已发布(4)/测试中(2)；
     * 已发布(4)→已停用(5)/已归档(6)；已停用(5)→已发布(4)/已归档(6)；已归档(6)→终态不可变更
     */
    private static final Map<Integer, Set<Integer>> ALLOWED_TRANSITIONS = Map.of(
            AlgorithmStatusEnum.DRAFT.getValue(), Set.of(AlgorithmStatusEnum.TESTING.getValue()),
            AlgorithmStatusEnum.TESTING.getValue(), Set.of(AlgorithmStatusEnum.PENDING_REVIEW.getValue(), AlgorithmStatusEnum.DRAFT.getValue()),
            AlgorithmStatusEnum.PENDING_REVIEW.getValue(), Set.of(AlgorithmStatusEnum.PUBLISHED.getValue(), AlgorithmStatusEnum.TESTING.getValue()),
            AlgorithmStatusEnum.PUBLISHED.getValue(), Set.of(AlgorithmStatusEnum.DISABLED.getValue(), AlgorithmStatusEnum.ARCHIVED.getValue()),
            AlgorithmStatusEnum.DISABLED.getValue(), Set.of(AlgorithmStatusEnum.PUBLISHED.getValue(), AlgorithmStatusEnum.ARCHIVED.getValue()),
            AlgorithmStatusEnum.ARCHIVED.getValue(), Set.of()
    );

    private void validateStatusTransition(Integer currentStatus, Integer targetStatus) {
        if (currentStatus == null) {
            return;
        }
        Set<Integer> allowed = ALLOWED_TRANSITIONS.get(currentStatus);
        if (allowed == null || !allowed.contains(targetStatus)) {
            throw new BusinessException(ResultCode.DATA_STATE_NOT_ALLOW,
                    "不允许将算法状态从 " + currentStatus + " 变更为 " + targetStatus);
        }
    }
}
