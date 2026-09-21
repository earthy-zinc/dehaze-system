package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
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
import com.pei.dehaze.model.form.AiEvalDatasetUpdateForm;
import com.pei.dehaze.model.form.AiEvalSampleCreateForm;
import com.pei.dehaze.model.form.AiEvalSampleUpdateForm;
import com.pei.dehaze.model.vo.AiEvalDatasetVO;
import com.pei.dehaze.model.vo.AiEvalRunVO;
import com.pei.dehaze.model.vo.AiEvalSampleVO;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 智能体评测服务（评测集/样本 CRUD、执行记录查询）。
 *
 * <p>对齐 dehaze-python {@code ai_eval_service} 的数据面语义：评测集软删方案 A（软删行复活）、
 * 样本随评测集物理清理、跨 Agent 操作一律 404。回归评测执行与发布 Agent（含门禁评测）
 * 需真实调用 LLM 判分，属转发域能力，由 dehaze-python 承接。
 *
 * @author dehaze
 */
@Service
@RequiredArgsConstructor
public class AiEvalService {

    private final SysAiAgentEvalDatasetMapper datasetMapper;

    private final SysAiAgentEvalSampleMapper sampleMapper;

    private final SysAiAgentEvalRunMapper runMapper;

    private final AiAgentService agentService;

    private final AuditLogService auditLogService;

    @Transactional
    public AiEvalDatasetVO createDataset(Long agentId, AiEvalDatasetCreateForm form) {
        agentService.getOrThrow(agentId);
        SysAiAgentEvalDataset existing = datasetMapper.selectByAgentAndTypeIgnoringDeleted(agentId,
                form.getDatasetType());
        String description = form.getDescription() == null ? "" : form.getDescription();
        if (existing != null) {
            if (existing.getDeleted() == null || existing.getDeleted() == 0) {
                throw new BusinessException(ResultCode.DATA_EXISTS, "该 Agent 已存在同类型评测集");
            }
            // 软删行占用唯一键 (agent_id, dataset_type)，复活原行而非插入
            datasetMapper.revive(existing.getId(), form.getName(), description);
            existing.setName(form.getName());
            existing.setDescription(description);
            existing.setDeleted(0L);
            return toDatasetVO(existing);
        }
        SysAiAgentEvalDataset dataset = new SysAiAgentEvalDataset();
        dataset.setAgentId(agentId);
        dataset.setName(form.getName());
        dataset.setDescription(description);
        dataset.setDatasetType(form.getDatasetType());
        datasetMapper.insert(dataset);
        return toDatasetVO(dataset);
    }

    public List<AiEvalDatasetVO> listDatasets(Long agentId) {
        return datasetMapper.selectList(new LambdaQueryWrapper<SysAiAgentEvalDataset>()
                        .eq(SysAiAgentEvalDataset::getAgentId, agentId)
                        .orderByDesc(SysAiAgentEvalDataset::getId))
                .stream().map(this::toDatasetVO).toList();
    }

    @Transactional
    public AiEvalDatasetVO updateDataset(Long agentId, Long datasetId, AiEvalDatasetUpdateForm form) {
        SysAiAgentEvalDataset dataset = getDatasetOfAgent(agentId, datasetId);
        if (form.getName() != null) {
            dataset.setName(form.getName());
        }
        if (form.getDescription() != null) {
            dataset.setDescription(form.getDescription());
        }
        datasetMapper.updateById(dataset);
        return toDatasetVO(dataset);
    }

    @Transactional
    public void deleteDataset(Long agentId, Long datasetId, Long operatorId) {
        SysAiAgentEvalDataset dataset = getDatasetOfAgent(agentId, datasetId);
        // 样本随数据集管理：数据集软删同时级联清理样本（样本表无逻辑删除列）
        sampleMapper.deleteByDatasetIds(List.of(dataset.getId()));
        datasetMapper.softDeleteByIds(List.of(dataset.getId()));
        Map<String, Object> beforeValue = new LinkedHashMap<>();
        beforeValue.put("agent_id", agentId);
        beforeValue.put("name", dataset.getName());
        beforeValue.put("dataset_type", dataset.getDatasetType());
        auditLogService.recordAudit(operatorId, "ai_eval_dataset", dataset.getId(), "delete", "ai_eval",
                beforeValue, null, null, null);
    }

    @Transactional
    public AiEvalSampleVO createSample(Long agentId, Long datasetId, AiEvalSampleCreateForm form) {
        if (!datasetId.equals(form.getDatasetId())) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "样本所属评测集与路径不一致");
        }
        getDatasetOfAgent(agentId, datasetId);
        SysAiAgentEvalSample sample = new SysAiAgentEvalSample();
        sample.setDatasetId(datasetId);
        sample.setTaskGoal(form.getTaskGoal());
        sample.setAllowedInput(form.getAllowedInput());
        sample.setTools(form.getTools());
        sample.setExpectedProcess(form.getExpectedProcess());
        sample.setExpectedResult(form.getExpectedResult());
        sample.setForbiddenBehavior(form.getForbiddenBehavior());
        sample.setRiskLevel(form.getRiskLevel() == null ? "low" : form.getRiskLevel());
        sampleMapper.insert(sample);
        return toSampleVO(sample);
    }

    public List<AiEvalSampleVO> listSamples(Long agentId, Long datasetId) {
        getDatasetOfAgent(agentId, datasetId);
        return sampleMapper.selectList(new LambdaQueryWrapper<SysAiAgentEvalSample>()
                        .eq(SysAiAgentEvalSample::getDatasetId, datasetId)
                        .orderByAsc(SysAiAgentEvalSample::getId))
                .stream().map(this::toSampleVO).toList();
    }

    @Transactional
    public AiEvalSampleVO updateSample(Long agentId, Long sampleId, AiEvalSampleUpdateForm form) {
        SysAiAgentEvalSample sample = getSampleOfAgent(agentId, sampleId);
        if (form.getTaskGoal() != null) {
            sample.setTaskGoal(form.getTaskGoal());
        }
        if (form.getAllowedInput() != null) {
            sample.setAllowedInput(form.getAllowedInput());
        }
        if (form.getTools() != null) {
            sample.setTools(form.getTools());
        }
        if (form.getExpectedProcess() != null) {
            sample.setExpectedProcess(form.getExpectedProcess());
        }
        if (form.getExpectedResult() != null) {
            sample.setExpectedResult(form.getExpectedResult());
        }
        if (form.getForbiddenBehavior() != null) {
            sample.setForbiddenBehavior(form.getForbiddenBehavior());
        }
        if (form.getRiskLevel() != null) {
            sample.setRiskLevel(form.getRiskLevel());
        }
        sampleMapper.updateById(sample);
        return toSampleVO(sample);
    }

    @Transactional
    public void deleteSample(Long agentId, Long sampleId, Long operatorId) {
        SysAiAgentEvalSample sample = getSampleOfAgent(agentId, sampleId);
        sampleMapper.deleteById(sample.getId());
        Map<String, Object> beforeValue = new LinkedHashMap<>();
        beforeValue.put("dataset_id", sample.getDatasetId());
        beforeValue.put("task_goal", sample.getTaskGoal());
        auditLogService.recordAudit(operatorId, "ai_eval_sample", sample.getId(), "delete", "ai_eval",
                beforeValue, null, null, null);
    }

    public IPage<AiEvalRunVO> listRuns(Long agentId, int pageNum, int pageSize, Long datasetId) {
        Page<SysAiAgentEvalRun> page = new Page<>(pageNum, pageSize);
        IPage<SysAiAgentEvalRun> runPage = runMapper.selectRunPage(page, agentId, datasetId);
        Page<AiEvalRunVO> result = new Page<>(pageNum, pageSize, runPage.getTotal());
        result.setRecords(new ArrayList<>(runPage.getRecords().stream().map(this::toRunVO).toList()));
        return result;
    }

    /**
     * 评测集必须归属于路径中的 Agent（跨 Agent 操作一律 404，不暴露存在性）
     */
    private SysAiAgentEvalDataset getDatasetOfAgent(Long agentId, Long datasetId) {
        SysAiAgentEvalDataset dataset = datasetMapper.selectById(datasetId);
        if (dataset == null || !dataset.getAgentId().equals(agentId)) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "评测集不存在");
        }
        return dataset;
    }

    private SysAiAgentEvalSample getSampleOfAgent(Long agentId, Long sampleId) {
        SysAiAgentEvalSample sample = sampleMapper.selectById(sampleId);
        if (sample == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "评测样本不存在");
        }
        getDatasetOfAgent(agentId, sample.getDatasetId());
        return sample;
    }

    private AiEvalDatasetVO toDatasetVO(SysAiAgentEvalDataset dataset) {
        AiEvalDatasetVO vo = new AiEvalDatasetVO();
        vo.setId(dataset.getId());
        vo.setAgentId(dataset.getAgentId());
        vo.setName(dataset.getName());
        vo.setDescription(dataset.getDescription());
        vo.setDatasetType(dataset.getDatasetType());
        vo.setCreateTime(dataset.getCreateTime());
        return vo;
    }

    private AiEvalSampleVO toSampleVO(SysAiAgentEvalSample sample) {
        AiEvalSampleVO vo = new AiEvalSampleVO();
        vo.setId(sample.getId());
        vo.setDatasetId(sample.getDatasetId());
        vo.setTaskGoal(sample.getTaskGoal());
        vo.setAllowedInput(sample.getAllowedInput());
        vo.setTools(sample.getTools());
        vo.setExpectedProcess(sample.getExpectedProcess());
        vo.setExpectedResult(sample.getExpectedResult());
        vo.setForbiddenBehavior(sample.getForbiddenBehavior());
        vo.setRiskLevel(sample.getRiskLevel());
        vo.setCreateTime(sample.getCreateTime());
        return vo;
    }

    private AiEvalRunVO toRunVO(SysAiAgentEvalRun run) {
        AiEvalRunVO vo = new AiEvalRunVO();
        vo.setId(run.getId());
        vo.setAgentId(run.getAgentId());
        vo.setDatasetId(run.getDatasetId());
        vo.setTriggerType(run.getTriggerType());
        vo.setStatus(run.getStatus());
        vo.setScoreSummary(run.getScoreSummary());
        vo.setResults(run.getResults());
        vo.setCreateBy(run.getCreateBy());
        vo.setCreateTime(run.getCreateTime());
        return vo;
    }
}
