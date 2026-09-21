package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysAiAgentEvalDatasetMapper;
import com.pei.dehaze.mapper.SysAiAgentEvalRunMapper;
import com.pei.dehaze.mapper.SysAiAgentEvalSampleMapper;
import com.pei.dehaze.model.entity.SysAiAgentEvalDataset;
import com.pei.dehaze.model.entity.SysAiAgentEvalRun;
import com.pei.dehaze.model.entity.SysAiAgentEvalSample;
import com.pei.dehaze.model.form.AiEvalDatasetCreateForm;
import com.pei.dehaze.model.form.AiEvalSampleCreateForm;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * 智能体评测服务单测：跨 Agent 越权不可见、评测集软删行复活、样本随集级联清理、执行记录分页。
 *
 * <p>回归评测执行与发布门禁需真实调用 LLM 判分，已随发布链路移交 python 转发域，
 * java 侧只保留评测资产的数据面语义。
 */
@DisplayName("AiEvalService 评测资产")
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class AiEvalServiceTest {

    @Mock
    private SysAiAgentEvalDatasetMapper datasetMapper;
    @Mock
    private SysAiAgentEvalSampleMapper sampleMapper;
    @Mock
    private SysAiAgentEvalRunMapper runMapper;
    @Mock
    private AiAgentService agentService;
    @Mock
    private AuditLogService auditLogService;

    @InjectMocks
    private AiEvalService service;

    private SysAiAgentEvalDataset dataset(Long id, Long agentId) {
        SysAiAgentEvalDataset dataset = new SysAiAgentEvalDataset();
        dataset.setId(id);
        dataset.setAgentId(agentId);
        dataset.setName("回归集");
        dataset.setDatasetType("regression");
        return dataset;
    }

    @Test
    @DisplayName("创建评测集：同类型活跃评测集已存在报 A0501")
    void createDatasetRejectsDuplicateType() {
        SysAiAgentEvalDataset existing = dataset(5L, 1L);
        existing.setDeleted(0L);
        when(datasetMapper.selectByAgentAndTypeIgnoringDeleted(1L, "regression")).thenReturn(existing);

        AiEvalDatasetCreateForm form = new AiEvalDatasetCreateForm();
        form.setName("回归集");
        form.setDatasetType("regression");

        assertThat(assertThrows(BusinessException.class, () -> service.createDataset(1L, form)).getResultCode())
                .isEqualTo(ResultCode.DATA_EXISTS);
        verify(datasetMapper, never()).insert(any(SysAiAgentEvalDataset.class));
    }

    @Test
    @DisplayName("创建评测集：命中软删行则复活原行（唯一键含 deleted）")
    void createDatasetRevivesSoftDeletedRow() {
        SysAiAgentEvalDataset existing = dataset(5L, 1L);
        existing.setDeleted(5L);
        when(datasetMapper.selectByAgentAndTypeIgnoringDeleted(1L, "regression")).thenReturn(existing);

        AiEvalDatasetCreateForm form = new AiEvalDatasetCreateForm();
        form.setName("回归集(重建)");
        form.setDescription("说明");
        form.setDatasetType("regression");

        assertThat(service.createDataset(1L, form).getId()).isEqualTo(5L);
        assertThat(existing.getDeleted()).isZero();
        assertThat(existing.getName()).isEqualTo("回归集(重建)");
        verify(datasetMapper).revive(5L, "回归集(重建)", "说明");
        verify(datasetMapper, never()).insert(any(SysAiAgentEvalDataset.class));
    }

    @Test
    @DisplayName("跨 Agent 操作评测集一律 A0401（不暴露存在性）")
    void crossAgentDatasetIsNotFound() {
        when(datasetMapper.selectById(5L)).thenReturn(dataset(5L, 2L));

        assertThat(assertThrows(BusinessException.class, () -> service.deleteDataset(1L, 5L, 9L)).getResultCode())
                .isEqualTo(ResultCode.RESOURCE_NOT_FOUND);
        verify(datasetMapper, never()).softDeleteByIds(anyList());
    }

    @Test
    @DisplayName("删除评测集：样本随集物理清理（样本表无逻辑删除列）并留审计")
    void deleteDatasetCascadesSamples() {
        when(datasetMapper.selectById(5L)).thenReturn(dataset(5L, 1L));

        service.deleteDataset(1L, 5L, 9L);

        verify(sampleMapper).deleteByDatasetIds(List.of(5L));
        verify(datasetMapper).softDeleteByIds(List.of(5L));
        verify(auditLogService).recordAudit(eq(9L), eq("ai_eval_dataset"), eq(5L), eq("delete"), eq("ai_eval"),
                any(), isNull(), isNull(), isNull());
    }

    @Test
    @DisplayName("创建样本：body 中的 datasetId 与路径不一致报 A0400（防跨集写入）")
    void createSampleRejectsMismatchedDatasetId() {
        AiEvalSampleCreateForm form = new AiEvalSampleCreateForm();
        form.setDatasetId(6L);
        form.setTaskGoal("目标");

        assertThat(assertThrows(BusinessException.class, () -> service.createSample(1L, 5L, form)).getResultCode())
                .isEqualTo(ResultCode.PARAM_ERROR);
        verify(sampleMapper, never()).insert(any(SysAiAgentEvalSample.class));
    }

    @Test
    @DisplayName("创建样本：默认风险等级为 low")
    void createSampleDefaultsRiskLevel() {
        when(datasetMapper.selectById(5L)).thenReturn(dataset(5L, 1L));
        AiEvalSampleCreateForm form = new AiEvalSampleCreateForm();
        form.setDatasetId(5L);
        form.setTaskGoal("目标");

        assertThat(service.createSample(1L, 5L, form).getRiskLevel()).isEqualTo("low");
    }

    @Test
    @DisplayName("删除样本：物理删除并留审计")
    void deleteSamplePurgesAndAudits() {
        SysAiAgentEvalSample sample = new SysAiAgentEvalSample();
        sample.setId(7L);
        sample.setDatasetId(5L);
        sample.setTaskGoal("目标");
        when(sampleMapper.selectById(7L)).thenReturn(sample);
        when(datasetMapper.selectById(5L)).thenReturn(dataset(5L, 1L));

        service.deleteSample(1L, 7L, 9L);

        verify(sampleMapper).deleteById(7L);
        verify(auditLogService).recordAudit(eq(9L), eq("ai_eval_sample"), eq(7L), eq("delete"), eq("ai_eval"),
                any(), isNull(), isNull(), isNull());
    }

    @Test
    @DisplayName("执行记录列表：按 agent 与可选评测集过滤分页")
    void listRunsPaginatesByAgentAndDataset() {
        SysAiAgentEvalRun run = new SysAiAgentEvalRun();
        run.setId(3L);
        run.setAgentId(1L);
        run.setStatus(2);
        Page<SysAiAgentEvalRun> page = new Page<>(1, 10, 1);
        page.setRecords(List.of(run));
        when(runMapper.selectRunPage(any(Page.class), eq(1L), eq(5L))).thenReturn(page);

        assertThat(service.listRuns(1L, 1, 10, 5L).getRecords()).hasSize(1);
        verify(runMapper).selectRunPage(any(Page.class), eq(1L), eq(5L));
    }
}
