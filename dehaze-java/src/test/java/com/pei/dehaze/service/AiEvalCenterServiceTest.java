package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.conditions.Wrapper;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.AiInsightMapper;
import com.pei.dehaze.mapper.SysAiAgentEvalRunMapper;
import com.pei.dehaze.mapper.SysAiAgentEvalSampleMapper;
import com.pei.dehaze.mapper.SysAiAgentMapper;
import com.pei.dehaze.mapper.SysAiEvalReviewMapper;
import com.pei.dehaze.model.entity.SysAiAgent;
import com.pei.dehaze.model.entity.SysAiAgentEvalRun;
import com.pei.dehaze.model.entity.SysAiEvalReview;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.function.Executable;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * 评测中心聚合服务单测：判分一致性三态、总览门禁状态、复核详情字段映射、复核回填幂等。
 *
 * <p>判分一致性决定发布门禁是否放行（{@code drifted} 会暂停依赖判分的门禁），
 * 复核队列与详情是人工校准判分模型的唯一入口，字段映射错位会让校准结论失真。
 */
@DisplayName("AiEvalCenterService 评测中心聚合")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AiEvalCenterServiceTest {

    @Mock
    private SysAiAgentMapper agentMapper;
    @Mock
    private SysAiAgentEvalRunMapper runMapper;
    @Mock
    private SysAiAgentEvalSampleMapper sampleMapper;
    @Mock
    private SysAiEvalReviewMapper reviewMapper;
    @Mock
    private AiInsightMapper insightMapper;

    @InjectMocks
    private AiEvalCenterService service;

    @BeforeEach
    void setUp() {
        when(insightMapper.selectDictValue(eq("ai_eval"), anyString())).thenReturn(null);
        when(reviewMapper.listAll(anyInt())).thenReturn(List.of());
        when(runMapper.selectCompleted(any(), any(), any(), anyInt())).thenReturn(List.of());
    }

    private SysAiEvalReview review(Long id, int status, Integer agree) {
        SysAiEvalReview review = new SysAiEvalReview();
        review.setId(id);
        review.setRunId(11L);
        review.setSampleId(21L);
        review.setAgentId(1L);
        review.setJudgePassed(0);
        review.setRiskLevel("high");
        review.setStatus(status);
        review.setAgree(agree);
        review.setRemark("备注");
        review.setCreateTime(LocalDateTime.now());
        return review;
    }

    private SysAiAgentEvalRun run(Long id, Long agentId, int status) {
        SysAiAgentEvalRun run = new SysAiAgentEvalRun();
        run.setId(id);
        run.setAgentId(agentId);
        run.setStatus(status);
        run.setTriggerType("manual");
        run.setCreateTime(LocalDateTime.now());
        return run;
    }

    private void assertBizError(ResultCode expected, Executable action) {
        assertThat(assertThrows(BusinessException.class, action).getResultCode()).isEqualTo(expected);
    }

    @Test
    @DisplayName("判分状态：无人工复核数据时为 insufficient_data（不误判为漂移）")
    void judgeStatusInsufficientWithoutReviews() {
        var status = service.judgeStatus();

        assertThat(status.getConsistencyState()).isEqualTo("insufficient_data");
        assertThat(status.getDriftPaused()).isFalse();
        assertThat(status.getConsistencyThreshold()).isEqualTo(90);
        assertThat(status.getReviewStats().getAgreementRate()).isEqualTo(0.0);
    }

    @Test
    @DisplayName("判分状态：一致率低于阈值判 drifted 并暂停门禁，达标判 normal")
    void judgeStatusDerivesFromAgreementRate() {
        when(reviewMapper.listAll(anyInt()))
                .thenReturn(List.of(review(1L, 2, 1), review(2L, 2, 0)));
        var drifted = service.judgeStatus();
        assertThat(drifted.getConsistencyState()).isEqualTo("drifted");
        assertThat(drifted.getDriftPaused()).isTrue();
        assertThat(drifted.getReviewStats().getAgreementRate()).isEqualTo(50.0);

        when(reviewMapper.listAll(anyInt())).thenReturn(List.of(review(1L, 2, 1)));
        var normal = service.judgeStatus();
        assertThat(normal.getConsistencyState()).isEqualTo("normal");
        assertThat(normal.getDriftPaused()).isFalse();
    }

    @Test
    @DisplayName("总览：无评测记录的 Agent 标记 gateStatus=none，有失败记录标记 failed")
    void overviewMarksGateStatus() {
        SysAiAgent withRun = new SysAiAgent();
        withRun.setId(1L);
        withRun.setAgentCode("a1");
        withRun.setName("Agent1");
        SysAiAgent withoutRun = new SysAiAgent();
        withoutRun.setId(2L);
        withoutRun.setAgentCode("a2");
        withoutRun.setName("Agent2");
        when(agentMapper.selectList(any(Wrapper.class))).thenReturn(List.of(withRun, withoutRun));
        when(runMapper.selectLatestPerAgent(2)).thenReturn(List.of(run(11L, 1L, 1)));

        var items = service.overview();

        assertThat(items).hasSize(2);
        assertThat(items.get(0).getGateStatus()).isEqualTo("failed");
        assertThat(items.get(0).getRunId()).isEqualTo(11L);
        assertThat(items.get(1).getGateStatus()).isEqualTo("none");
        assertThat(items.get(1).getTotalScore()).isNull();
        assertThat(items.get(1).getDegraded()).isFalse();
    }

    @Test
    @DisplayName("总览：高风险失败样本会被标注（安全类回归的关键信号）")
    void overviewFlagsHighRiskFailure() {
        when(agentMapper.selectList(any(Wrapper.class))).thenReturn(List.of(agent(1L)));
        SysAiAgentEvalRun latest = run(11L, 1L, 2);
        latest.setResults(List.of(Map.of("sample_id", 21, "risk_level", "high", "passed", false)));
        when(runMapper.selectLatestPerAgent(2)).thenReturn(List.of(latest));

        assertThat(service.overview().get(0).getHighRiskFailed()).isTrue();
    }

    @Test
    @DisplayName("复核详情：评测记录或样本结果缺失报 A0401（不回退成空详情）")
    void reviewDetailRequiresRunAndResult() {
        when(runMapper.selectById(11L)).thenReturn(null);
        assertBizError(ResultCode.RESOURCE_NOT_FOUND, () -> service.reviewDetail(11L, 21L));

        when(runMapper.selectById(11L)).thenReturn(run(11L, 1L, 2));
        assertBizError(ResultCode.RESOURCE_NOT_FOUND, () -> service.reviewDetail(11L, 21L));
    }

    @Test
    @DisplayName("复核详情：样本已删除仍可复核历史结果，字段优先取评测结果快照")
    void reviewDetailFallsBackToRunSnapshot() {
        SysAiAgentEvalRun run = run(11L, 1L, 2);
        run.setResults(List.of(Map.of(
                "sample_id", 21,
                "task_goal", "清理图片",
                "risk_level", "high",
                "passed", false,
                "actual_output", "实际输出",
                "scores", Map.of("result_quality", 3),
                "notes", Map.of("result_quality", "细节缺失"))));
        when(runMapper.selectById(11L)).thenReturn(run);
        when(sampleMapper.selectById(21L)).thenReturn(null);
        when(agentMapper.selectById(1L)).thenReturn(agent(1L));

        var detail = service.reviewDetail(11L, 21L);

        assertThat(detail.getTaskGoal()).isEqualTo("清理图片");
        assertThat(detail.getRiskLevel()).isEqualTo("high");
        assertThat(detail.getJudgePassed()).isFalse();
        assertThat(detail.getActualOutput()).isEqualTo("实际输出");
        assertThat(detail.getScores()).containsEntry("result_quality", 3);
        assertThat(detail.getAgentName()).isEqualTo("Agent1");
    }

    @Test
    @DisplayName("复核回填：已完成复核不允许重复回填（A0503）")
    void submitReviewRejectsSecondSubmit() {
        when(reviewMapper.selectById(31L)).thenReturn(review(31L, 2, 1));

        assertBizError(ResultCode.OPERATION_NOT_ALLOW,
                () -> service.submitReview(31L, true, "再次复核", 9L));
    }

    @Test
    @DisplayName("复核回填：写入复核结论并返回 snake_case 结果键")
    void submitReviewPersistsConclusion() {
        SysAiEvalReview pending = review(31L, 1, null);
        when(reviewMapper.selectById(31L)).thenReturn(pending);

        Map<String, Object> result = service.submitReview(31L, false, "判分偏松", 9L);

        assertThat(pending.getStatus()).isEqualTo(2);
        assertThat(pending.getAgree()).isZero();
        assertThat(pending.getReviewerId()).isEqualTo(9L);
        assertThat(pending.getRemark()).isEqualTo("判分偏松");
        verify(reviewMapper).updateById(pending);
        assertThat(result).containsKeys("id", "run_id", "sample_id", "judge_passed", "agree", "remark");
        assertThat(result.get("agree")).isEqualTo(false);
    }

    @Test
    @DisplayName("复核队列：按状态过滤并统计待复核/已复核数量")
    void listReviewsCountsByStatus() {
        when(reviewMapper.listAll(anyInt()))
                .thenReturn(List.of(review(1L, 1, null), review(2L, 2, 1)));
        when(agentMapper.selectList(any(Wrapper.class))).thenReturn(List.of(agent(1L)));

        var queue = service.listReviews(null);

        assertThat(queue.getPending()).isEqualTo(1);
        assertThat(queue.getReviewed()).isEqualTo(1);
        assertThat(queue.getItems()).hasSize(2);

        when(reviewMapper.listAll(anyInt())).thenReturn(List.of(review(1L, 1, null)));
        assertThat(service.listReviews(1).getItems()).hasSize(1);
    }

    private SysAiAgent agent(Long id) {
        SysAiAgent agent = new SysAiAgent();
        agent.setId(id);
        agent.setAgentCode("a" + id);
        agent.setName("Agent" + id);
        return agent;
    }
}
