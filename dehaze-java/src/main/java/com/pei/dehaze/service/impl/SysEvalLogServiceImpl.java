package com.pei.dehaze.service.impl;

import cn.hutool.core.bean.BeanUtil;
import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.enums.LogStatusEnum;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysEvalLogMapper;
import com.pei.dehaze.model.entity.SysAlgorithm;
import com.pei.dehaze.model.entity.SysEvalLog;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.model.form.EvaluationForm;
import com.pei.dehaze.model.query.EvalLogQuery;
import com.pei.dehaze.model.vo.EvalLogVO;
import com.pei.dehaze.model.vo.EvaluationResultVO;
import com.pei.dehaze.service.SysAlgorithmService;
import com.pei.dehaze.service.SysEvalLogService;
import com.pei.dehaze.service.SysFileService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class SysEvalLogServiceImpl extends ServiceImpl<SysEvalLogMapper, SysEvalLog> implements SysEvalLogService {

    private final SysAlgorithmService algorithmService;
    private final SysFileService sysFileService;
    private final EvalLogAsyncTask asyncTask;

    @Value("${file.datasetBaseUrl}")
    private String datasetBaseUrl;

    @Override
    public EvaluationResultVO evaluate(EvaluationForm form) {
        SysAlgorithm algorithm = algorithmService.getById(form.getAlgorithmId());
        if (algorithm == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "算法不存在");
        }

        String predUrl = toAbsoluteUrl(form.getPredUrl() != null ? form.getPredUrl() : resolveFileUrl(form.getPredFileId(), "pred"));
        String gtUrl = toAbsoluteUrl(form.getGtUrl() != null ? form.getGtUrl() : resolveFileUrl(form.getGtFileId(), "gt"));

        SysEvalLog evalLog = new SysEvalLog();
        evalLog.setAlgorithmId(form.getAlgorithmId());
        if (form.getPredFileId() != null) {
            evalLog.setPredFileId(form.getPredFileId());
        }
        if (form.getGtFileId() != null) {
            evalLog.setGtFileId(form.getGtFileId());
        }
        evalLog.setPredUrl(predUrl);
        evalLog.setGtUrl(gtUrl);
        evalLog.setStatus(LogStatusEnum.PROCESSING);
        this.save(evalLog);

        asyncTask.execute(evalLog.getId(), form.getAlgorithmId(), predUrl, gtUrl);

        EvaluationResultVO vo = new EvaluationResultVO();
        vo.setLogId(evalLog.getId());
        vo.setStatus(LogStatusEnum.PROCESSING);
        return vo;
    }

    @Override
    public Page<EvalLogVO> getEvalLogPage(EvalLogQuery query) {
        Page<SysEvalLog> page = new Page<>(query.getPageNum(), query.getPageSize());
        LambdaQueryWrapper<SysEvalLog> wrapper = new LambdaQueryWrapper<SysEvalLog>()
                .eq(query.getAlgorithmId() != null, SysEvalLog::getAlgorithmId, query.getAlgorithmId())
                .orderByDesc(SysEvalLog::getCreateTime);

        Page<SysEvalLog> result = this.page(page, wrapper);
        Page<EvalLogVO> voPage = new Page<>(result.getCurrent(), result.getSize(), result.getTotal());

        Set<Long> algorithmIds = result.getRecords().stream()
                .map(SysEvalLog::getAlgorithmId)
                .filter(Objects::nonNull)
                .collect(Collectors.toSet());
        Map<Long, String> algorithmNameMap = algorithmIds.isEmpty()
                ? Collections.emptyMap()
                : algorithmService.listByIds(algorithmIds).stream()
                        .collect(Collectors.toMap(SysAlgorithm::getId, SysAlgorithm::getName, (a, b) -> a));

        voPage.setRecords(result.getRecords().stream().map(log -> {
            EvalLogVO vo = new EvalLogVO();
            BeanUtil.copyProperties(log, vo);
            vo.setAlgorithmName(algorithmNameMap.getOrDefault(log.getAlgorithmId(), "未知算法"));
            return vo;
        }).toList());
        return voPage;
    }

    private String resolveFileUrl(Long fileId, String type) {
        if (fileId != null) {
            SysFile sysFile = sysFileService.getById(fileId);
            if (sysFile != null && CharSequenceUtil.isNotBlank(sysFile.getUrl())) {
                return sysFile.getUrl();
            }
            log.warn("文件不存在或 URL 为空: fileId={}", fileId);
        }
        throw new BusinessException(ResultCode.PARAM_ERROR, "缺少" + ("pred".equals(type) ? "预测" : "参考") + "图片");
    }

    private String toAbsoluteUrl(String url) {
        if (CharSequenceUtil.isBlank(url)) {
            return url;
        }
        if (url.startsWith("http://") || url.startsWith("https://")) {
            return url;
        }
        if (url.startsWith("/dataset-api/")) {
            return datasetBaseUrl + url.substring("/dataset-api".length());
        }
        return url;
    }
}
