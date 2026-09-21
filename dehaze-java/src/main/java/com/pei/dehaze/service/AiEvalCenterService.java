package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.AiInsightMapper;
import com.pei.dehaze.mapper.SysAiAgentEvalRunMapper;
import com.pei.dehaze.mapper.SysAiAgentEvalSampleMapper;
import com.pei.dehaze.mapper.SysAiAgentMapper;
import com.pei.dehaze.mapper.SysAiEvalReviewMapper;
import com.pei.dehaze.model.entity.SysAiAgent;
import com.pei.dehaze.model.entity.SysAiAgentEvalRun;
import com.pei.dehaze.model.entity.SysAiAgentEvalSample;
import com.pei.dehaze.model.entity.SysAiEvalReview;
import com.pei.dehaze.model.vo.AiEvalCompareVO;
import com.pei.dehaze.model.vo.AiEvalOverviewVO;
import com.pei.dehaze.model.vo.AiEvalReviewDetailVO;
import com.pei.dehaze.model.vo.AiEvalReviewItemVO;
import com.pei.dehaze.model.vo.AiEvalReviewQueueVO;
import com.pei.dehaze.model.vo.AiEvalRunSnapshotVO;
import com.pei.dehaze.model.vo.AiEvalSampleDiffItemVO;
import com.pei.dehaze.model.vo.AiEvalSampleDiffVO;
import com.pei.dehaze.model.vo.AiEvalTrendVO;
import com.pei.dehaze.model.vo.AiJudgeReviewStatsVO;
import com.pei.dehaze.model.vo.AiJudgeStatusVO;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 评测中心聚合服务（跨 Agent 总览/趋势/对比/判分状态/人工复核）。
 *
 * <p>对齐 dehaze-python {@code ai_eval_center_service}：退化判定相对上次完成评测总分下降超阈值；
 * 判分一致性由人工复核一致率推导；复核队列按"失败样本全量 + 通过样本确定性抽样"幂等补齐。
 *
 * @author dehaze
 */
@Service
@RequiredArgsConstructor
public class AiEvalCenterService {

    private static final String DICT_TYPE_AI_EVAL = "ai_eval";

    private static final int REGRESSION_THRESHOLD_DEFAULT = 5;

    private static final int CONSISTENCY_THRESHOLD_DEFAULT = 90;

    private static final int REVIEW_RATIO_DEFAULT = 1;

    private static final int REVIEW_SCAN_RUN_LIMIT = 50;

    private static final int REVIEW_AGGREGATE_LIMIT = 1000;

    private static final List<String> DIMENSIONS = List.of("result_quality", "process_compliance",
            "safety_boundary", "efficiency");

    private final SysAiAgentMapper agentMapper;

    private final SysAiAgentEvalRunMapper runMapper;

    private final SysAiAgentEvalSampleMapper sampleMapper;

    private final SysAiEvalReviewMapper reviewMapper;

    private final AiInsightMapper insightMapper;

    /**
     * 判分模型状态：由人工复核一致率与阈值推导（漂移仅暂停依赖判分的门禁判定）
     */
    public AiJudgeStatusVO judgeStatus() {
        int threshold = dictInt("judge_consistency_threshold", CONSISTENCY_THRESHOLD_DEFAULT);
        List<SysAiEvalReview> reviews = reviewMapper.listAll(REVIEW_AGGREGATE_LIMIT);
        List<SysAiEvalReview> reviewed = reviews.stream()
                .filter(r -> Integer.valueOf(2).equals(r.getStatus())).toList();
        int agreeCount = (int) reviewed.stream().filter(r -> Integer.valueOf(1).equals(r.getAgree())).count();
        int disagreeCount = reviewed.size() - agreeCount;
        AiJudgeReviewStatsVO stats = new AiJudgeReviewStatsVO();
        stats.setTotal(reviews.size());
        stats.setPending(reviews.size() - reviewed.size());
        stats.setReviewed(reviewed.size());
        stats.setAgreeCount(agreeCount);
        stats.setDisagreeCount(disagreeCount);
        stats.setAgreementRate(reviewed.isEmpty() ? 0.0
                : round(agreeCount * 100.0 / reviewed.size(), 2));
        String state;
        if (reviewed.isEmpty()) {
            state = "insufficient_data";
        } else {
            state = stats.getAgreementRate() >= threshold ? "normal" : "drifted";
        }
        AiJudgeStatusVO vo = new AiJudgeStatusVO();
        vo.setConsistencyState(state);
        vo.setDriftPaused("drifted".equals(state));
        vo.setConsistencyThreshold(threshold);
        vo.setReviewStats(stats);
        return vo;
    }

    /**
     * 评测总览：各 Agent 最近两次已完成评测（门禁状态/总分/退化/高风险失败）
     */
    public List<AiEvalOverviewVO> overview() {
        List<SysAiAgent> agents = agentMapper.selectList(new LambdaQueryWrapper<>());
        List<SysAiAgentEvalRun> runs = runMapper.selectLatestPerAgent(2);
        Map<Long, List<SysAiAgentEvalRun>> runsByAgent = new HashMap<>();
        for (SysAiAgentEvalRun run : runs) {
            runsByAgent.computeIfAbsent(run.getAgentId(), k -> new ArrayList<>()).add(run);
        }
        int threshold = dictInt("regression_threshold", REGRESSION_THRESHOLD_DEFAULT);
        List<AiEvalOverviewVO> items = new ArrayList<>();
        for (SysAiAgent agent : agents) {
            List<SysAiAgentEvalRun> agentRuns = runsByAgent.getOrDefault(agent.getId(), List.of());
            SysAiAgentEvalRun latest = agentRuns.isEmpty() ? null : agentRuns.get(0);
            SysAiAgentEvalRun previous = agentRuns.size() > 1 ? agentRuns.get(1) : null;
            Double total = latest == null ? null : totalScore(latest.getScoreSummary());
            AiEvalOverviewVO item = new AiEvalOverviewVO();
            item.setAgentId(agent.getId());
            item.setAgentCode(agent.getAgentCode());
            item.setAgentName(agent.getName());
            item.setRunId(latest == null ? null : latest.getId());
            item.setRunTime(latest == null ? null : latest.getCreateTime());
            item.setTriggerType(latest == null ? null : latest.getTriggerType());
            item.setGateStatus(latest == null ? "none"
                    : (Integer.valueOf(2).equals(latest.getStatus()) ? "passed" : "failed"));
            item.setTotalScore(total);
            item.setDimensions(latest == null ? null
                    : (Map<String, Object>) (latest.getScoreSummary() == null ? Map.of()
                    : latest.getScoreSummary().get("dimensions")));
            item.setDegraded(isDegraded(total,
                    previous == null ? null : totalScore(previous.getScoreSummary()), threshold));
            item.setHighRiskFailed(hasHighRiskFailed(latest));
            items.add(item);
        }
        return items;
    }

    /**
     * 历史趋势：已完成评测按时间升序（可按 Agent 与时间范围过滤）
     */
    public List<AiEvalTrendVO> trends(Long agentId, LocalDateTime startTime, LocalDateTime endTime, int limit) {
        List<SysAiAgentEvalRun> runs = runMapper.selectCompleted(agentId, startTime, endTime, limit);
        Map<Long, String> agentNames = agentNames(runs.stream().map(SysAiAgentEvalRun::getAgentId).toList());
        List<AiEvalTrendVO> items = new ArrayList<>();
        for (SysAiAgentEvalRun run : runs) {
            AiEvalTrendVO item = new AiEvalTrendVO();
            item.setRunId(run.getId());
            item.setAgentId(run.getAgentId());
            item.setAgentName(agentNames.get(run.getAgentId()));
            item.setTriggerType(run.getTriggerType());
            item.setStatus(run.getStatus());
            item.setTotalScore(totalScore(run.getScoreSummary()));
            item.setDimensions(run.getScoreSummary() == null ? null
                    : (Map<String, Object>) run.getScoreSummary().get("dimensions"));
            item.setCreateTime(run.getCreateTime());
            items.add(item);
        }
        return items;
    }

    /**
     * 两次评测对比：得分快照 + 四维差异 + 样本级差异（跨 Agent 对比拒绝）
     */
    public AiEvalCompareVO compareRuns(Long runId, Long baseRunId) {
        SysAiAgentEvalRun run = runMapper.selectById(runId);
        SysAiAgentEvalRun base = runMapper.selectById(baseRunId);
        if (run == null || base == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "评测记录不存在");
        }
        if (!run.getAgentId().equals(base.getAgentId())) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "两次评测不属于同一 Agent，无法对比");
        }
        AiEvalCompareVO vo = new AiEvalCompareVO();
        vo.setRunId(run.getId());
        vo.setBaseRunId(base.getId());
        vo.setAgentId(run.getAgentId());
        vo.setCurrent(runSnapshot(run));
        vo.setBase(runSnapshot(base));
        Map<String, Object> dimensionDiff = new LinkedHashMap<>();
        Map<String, Object> currentDims = dimensions(run.getScoreSummary());
        Map<String, Object> baseDims = dimensions(base.getScoreSummary());
        for (String dim : DIMENSIONS) {
            dimensionDiff.put(dim, round(toDouble(currentDims.get(dim)) - toDouble(baseDims.get(dim)), 2));
        }
        vo.setDimensionDiff(dimensionDiff);
        vo.setSampleDiff(sampleDiff(run.getResults(), base.getResults()));
        return vo;
    }

    /**
     * 复核队列：先按抽样规则幂等补齐最近评测的待复核项，再返回队列与统计
     */
    @Transactional
    public AiEvalReviewQueueVO listReviews(Integer status) {
        List<SysAiAgentEvalRun> runs = runMapper.selectCompleted(null, null, null, REVIEW_SCAN_RUN_LIMIT);
        materializeReviews(runs);
        List<SysAiEvalReview> reviews = reviewMapper.listAll(REVIEW_AGGREGATE_LIMIT);
        if (status != null) {
            reviews = reviews.stream().filter(r -> status.equals(r.getStatus())).toList();
        }
        Map<Long, String> agentNames = agentNames(reviews.stream().map(SysAiEvalReview::getAgentId).toList());
        List<AiEvalReviewItemVO> items = new ArrayList<>();
        for (SysAiEvalReview review : reviews) {
            AiEvalReviewItemVO item = new AiEvalReviewItemVO();
            item.setId(review.getId());
            item.setRunId(review.getRunId());
            item.setSampleId(review.getSampleId());
            item.setAgentId(review.getAgentId());
            item.setAgentName(agentNames.get(review.getAgentId()));
            item.setJudgePassed(Integer.valueOf(1).equals(review.getJudgePassed()));
            item.setRiskLevel(review.getRiskLevel());
            item.setStatus(review.getStatus());
            item.setAgree(review.getAgree() == null ? null : Integer.valueOf(1).equals(review.getAgree()));
            item.setRemark(review.getRemark());
            item.setCreateTime(review.getCreateTime());
            items.add(item);
        }
        AiEvalReviewQueueVO vo = new AiEvalReviewQueueVO();
        vo.setItems(items);
        vo.setPending((int) reviews.stream().filter(r -> Integer.valueOf(1).equals(r.getStatus())).count());
        vo.setReviewed((int) reviews.stream().filter(r -> Integer.valueOf(2).equals(r.getStatus())).count());
        return vo;
    }

    /**
     * 复核详情：样本定义 + 本次实际输出 + 四维得分与说明（样本随数据集删除后仍可复核历史结果）
     */
    public AiEvalReviewDetailVO reviewDetail(Long runId, Long sampleId) {
        SysAiAgentEvalRun run = runMapper.selectById(runId);
        if (run == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "评测记录不存在");
        }
        Map<String, Object> result = findResult(run, sampleId);
        if (result == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "该评测记录中没有此样本的执行结果");
        }
        SysAiAgentEvalSample sample = sampleMapper.selectById(sampleId);
        SysAiAgent agent = agentMapper.selectById(run.getAgentId());
        AiEvalReviewDetailVO vo = new AiEvalReviewDetailVO();
        vo.setRunId(run.getId());
        vo.setAgentId(run.getAgentId());
        vo.setAgentName(agent == null ? null : agent.getName());
        vo.setSampleId(sampleId);
        vo.setTaskGoal(result.get("task_goal") != null ? String.valueOf(result.get("task_goal"))
                : (sample == null ? "" : sample.getTaskGoal()));
        vo.setAllowedInput(sample == null ? null : sample.getAllowedInput());
        vo.setExpectedResult(sample == null ? null : sample.getExpectedResult());
        vo.setExpectedProcess(sample == null ? null : sample.getExpectedProcess());
        vo.setForbiddenBehavior(sample == null ? null : sample.getForbiddenBehavior());
        vo.setTools(sample == null ? null : sample.getTools());
        vo.setRiskLevel(result.get("risk_level") != null ? String.valueOf(result.get("risk_level"))
                : (sample == null ? "low" : sample.getRiskLevel()));
        vo.setJudgePassed(Boolean.TRUE.equals(result.get("passed")));
        vo.setActualOutput(result.get("actual_output") == null ? null : String.valueOf(result.get("actual_output")));
        vo.setError(result.get("error") == null ? null : String.valueOf(result.get("error")));
        vo.setScores(asStringMap(result.get("scores")));
        vo.setNotes(asStringMap(result.get("notes")));
        return vo;
    }

    /**
     * 复核结果回填（判定一致/不一致 + 备注），已完成复核不允许重复回填
     */
    @Transactional
    public Map<String, Object> submitReview(Long reviewId, Boolean agree, String remark, Long reviewerId) {
        SysAiEvalReview review = reviewMapper.selectById(reviewId);
        if (review == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "复核项不存在");
        }
        if (Integer.valueOf(2).equals(review.getStatus())) {
            throw new BusinessException(ResultCode.OPERATION_NOT_ALLOW, "该复核项已完成复核，不允许重复回填");
        }
        review.setAgree(Boolean.TRUE.equals(agree) ? 1 : 0);
        review.setStatus(2);
        review.setReviewerId(reviewerId);
        review.setRemark(remark);
        reviewMapper.updateById(review);
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("id", review.getId());
        result.put("run_id", review.getRunId());
        result.put("sample_id", review.getSampleId());
        result.put("agent_id", review.getAgentId());
        result.put("judge_passed", Integer.valueOf(1).equals(review.getJudgePassed()));
        result.put("risk_level", review.getRiskLevel());
        result.put("status", review.getStatus());
        result.put("agree", Boolean.TRUE.equals(agree));
        result.put("remark", review.getRemark());
        return result;
    }

    /**
     * 按抽样规则为最近评测生成待复核项：失败样本全量 + 通过样本按比例确定性抽样，(run_id, sample_id) 幂等
     */
    private void materializeReviews(List<SysAiAgentEvalRun> runs) {
        if (runs.isEmpty()) {
            return;
        }
        int ratio = dictInt("judge_review_ratio", REVIEW_RATIO_DEFAULT);
        Set<String> existing = new HashSet<>();
        for (SysAiEvalReview review : reviewMapper.listByRunIds(runs.stream()
                .map(SysAiAgentEvalRun::getId).toList())) {
            existing.add(review.getRunId() + ":" + review.getSampleId());
        }
        for (SysAiAgentEvalRun run : runs) {
            for (Map<String, Object> result : resultsOf(run)) {
                Object sampleIdValue = result.get("sample_id");
                if (!(sampleIdValue instanceof Number sampleIdNumber)) {
                    continue;
                }
                Long sampleId = sampleIdNumber.longValue();
                boolean passed = Boolean.TRUE.equals(result.get("passed"));
                if (passed && !sampleHit(run.getId(), sampleId, ratio)) {
                    continue;
                }
                if (!existing.add(run.getId() + ":" + sampleId)) {
                    continue;
                }
                SysAiEvalReview review = new SysAiEvalReview();
                review.setRunId(run.getId());
                review.setSampleId(sampleId);
                review.setAgentId(run.getAgentId());
                review.setJudgePassed(passed ? 1 : 0);
                review.setRiskLevel(result.get("risk_level") == null ? "low"
                        : String.valueOf(result.get("risk_level")));
                review.setStatus(1);
                reviewMapper.insert(review);
            }
        }
    }

    /**
     * 确定性抽样：ratio 为百分比，同一 (run_id, sample_id) 结果恒定
     */
    private boolean sampleHit(Long runId, Long sampleId, int ratio) {
        return (runId * 1000003L + sampleId) % 100 < ratio;
    }

    private int dictInt(String name, int defaultValue) {
        String value = insightMapper.selectDictValue(DICT_TYPE_AI_EVAL, name);
        if (value == null || value.isBlank()) {
            return defaultValue;
        }
        try {
            return Integer.parseInt(value.trim());
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    private Map<Long, String> agentNames(List<Long> agentIds) {
        List<Long> ids = agentIds.stream().distinct().toList();
        if (ids.isEmpty()) {
            return Map.of();
        }
        Map<Long, String> names = new HashMap<>();
        for (SysAiAgent agent : agentMapper.selectList(
                new LambdaQueryWrapper<SysAiAgent>().in(SysAiAgent::getId, ids))) {
            names.put(agent.getId(), agent.getName());
        }
        return names;
    }

    private Map<String, Object> findResult(SysAiAgentEvalRun run, Long sampleId) {
        for (Map<String, Object> result : resultsOf(run)) {
            Object value = result.get("sample_id");
            if (value instanceof Number number && number.longValue() == sampleId) {
                return result;
            }
        }
        return null;
    }

    private List<Map<String, Object>> resultsOf(SysAiAgentEvalRun run) {
        return run.getResults() == null ? List.of() : run.getResults();
    }

    private AiEvalRunSnapshotVO runSnapshot(SysAiAgentEvalRun run) {
        Map<String, Object> summary = run.getScoreSummary() == null ? Map.of() : run.getScoreSummary();
        AiEvalRunSnapshotVO vo = new AiEvalRunSnapshotVO();
        vo.setRunId(run.getId());
        vo.setTotalScore(totalScore(summary));
        vo.setDimensions(dimensions(summary));
        Object count = summary.get("sample_count");
        vo.setSampleCount(count instanceof Number number ? number.intValue() : 0);
        Object passRate = summary.get("pass_rate");
        vo.setPassRate(passRate instanceof Number number ? number.doubleValue() : null);
        vo.setCreateTime(run.getCreateTime());
        return vo;
    }

    private AiEvalSampleDiffVO sampleDiff(List<Map<String, Object>> current, List<Map<String, Object>> base) {
        Map<Long, Map<String, Object>> curMap = bySampleId(current);
        Map<Long, Map<String, Object>> baseMap = bySampleId(base);
        List<AiEvalSampleDiffItemVO> added = new ArrayList<>();
        List<AiEvalSampleDiffItemVO> removed = new ArrayList<>();
        List<AiEvalSampleDiffItemVO> changed = new ArrayList<>();
        int unchanged = 0;
        for (Map.Entry<Long, Map<String, Object>> entry : curMap.entrySet()) {
            if (!baseMap.containsKey(entry.getKey())) {
                added.add(diffItem(entry.getKey(), entry.getValue(), null));
                continue;
            }
            Map<String, Object> baseResult = baseMap.get(entry.getKey());
            Double curTotal = sampleTotal(entry.getValue());
            Double baseTotal = sampleTotal(baseResult);
            if (!java.util.Objects.equals(entry.getValue().get("passed"), baseResult.get("passed"))
                    || !java.util.Objects.equals(curTotal, baseTotal)) {
                changed.add(diffItem(entry.getKey(), entry.getValue(), baseResult));
            } else {
                unchanged++;
            }
        }
        for (Map.Entry<Long, Map<String, Object>> entry : baseMap.entrySet()) {
            if (!curMap.containsKey(entry.getKey())) {
                removed.add(diffItem(entry.getKey(), entry.getValue(), null));
            }
        }
        AiEvalSampleDiffVO vo = new AiEvalSampleDiffVO();
        vo.setAdded(added);
        vo.setRemoved(removed);
        vo.setChanged(changed);
        vo.setUnchangedCount(unchanged);
        return vo;
    }

    private AiEvalSampleDiffItemVO diffItem(Long sampleId, Map<String, Object> result,
                                            Map<String, Object> baseResult) {
        AiEvalSampleDiffItemVO item = new AiEvalSampleDiffItemVO();
        item.setSampleId(sampleId);
        item.setTaskGoal(result.get("task_goal") == null ? "" : String.valueOf(result.get("task_goal")));
        item.setCurrentPassed(Boolean.TRUE.equals(result.get("passed")));
        item.setBasePassed(baseResult == null ? null : Boolean.TRUE.equals(baseResult.get("passed")));
        item.setCurrentScore(sampleTotal(result));
        item.setBaseScore(baseResult == null ? null : sampleTotal(baseResult));
        item.setScoreDelta(baseResult == null ? null
                : round(sampleTotal(result) - sampleTotal(baseResult), 2));
        return item;
    }

    private Map<Long, Map<String, Object>> bySampleId(List<Map<String, Object>> results) {
        Map<Long, Map<String, Object>> map = new LinkedHashMap<>();
        for (Map<String, Object> result : results == null ? List.<Map<String, Object>>of() : results) {
            Object value = result.get("sample_id");
            if (value instanceof Number number) {
                map.put(number.longValue(), result);
            }
        }
        return map;
    }

    private Double sampleTotal(Map<String, Object> result) {
        if (result == null) {
            return null;
        }
        Map<String, Object> scores = asStringMap(result.get("scores"));
        if (scores.isEmpty()) {
            return null;
        }
        double sum = scores.values().stream().mapToDouble(this::toDouble).sum();
        return round(sum / scores.size(), 2);
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> dimensions(Map<String, Object> summary) {
        Object value = summary == null ? null : summary.get("dimensions");
        return value instanceof Map ? (Map<String, Object>) value : null;
    }

    private Double totalScore(Map<String, Object> summary) {
        Map<String, Object> dims = dimensions(summary);
        if (dims == null || dims.isEmpty()) {
            return null;
        }
        double sum = dims.values().stream().mapToDouble(this::toDouble).sum();
        return round(sum / dims.size(), 2);
    }

    /**
     * 退化判定：相对上次评测总分下降超过阈值（百分比）
     */
    private boolean isDegraded(Double current, Double previous, int threshold) {
        if (current == null || previous == null || previous <= 0) {
            return false;
        }
        return (previous - current) / previous * 100 > threshold;
    }

    private boolean hasHighRiskFailed(SysAiAgentEvalRun run) {
        if (run == null) {
            return false;
        }
        for (Map<String, Object> result : resultsOf(run)) {
            if ("high".equals(result.get("risk_level")) && !Boolean.TRUE.equals(result.get("passed"))) {
                return true;
            }
        }
        return false;
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> asStringMap(Object value) {
        return value instanceof Map ? (Map<String, Object>) value : Map.of();
    }

    private double toDouble(Object value) {
        return value instanceof Number number ? number.doubleValue() : 0.0;
    }

    private double round(double value, int scale) {
        return new java.math.BigDecimal(value).setScale(scale, java.math.RoundingMode.HALF_UP).doubleValue();
    }
}
