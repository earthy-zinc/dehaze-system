package com.pei.dehaze.service.impl;

import cn.hutool.core.bean.BeanUtil;
import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysPredLogMapper;
import com.pei.dehaze.model.entity.SysAlgorithm;
import com.pei.dehaze.model.entity.SysPredLog;
import com.pei.dehaze.model.form.PredictionForm;
import com.pei.dehaze.model.query.PredLogQuery;
import com.pei.dehaze.model.vo.PredLogVO;
import com.pei.dehaze.model.vo.PredictionResultVO;
import com.pei.dehaze.service.SysAlgorithmService;
import com.pei.dehaze.service.SysFileService;
import com.pei.dehaze.service.SysPredLogService;
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
public class SysPredLogServiceImpl extends ServiceImpl<SysPredLogMapper, SysPredLog> implements SysPredLogService {

    private final SysAlgorithmService algorithmService;
    private final SysFileService sysFileService;
    private final PredLogAsyncTask asyncTask;

    @Value("${file.datasetBaseUrl}")
    private String datasetBaseUrl;

    @Override
    public PredictionResultVO predict(PredictionForm form) {
        SysAlgorithm algorithm = algorithmService.getById(form.getAlgorithmId());
        if (algorithm == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "算法不存在");
        }

        String imageUrl = resolveImageUrl(form);
        if (CharSequenceUtil.isBlank(imageUrl)) {
            throw new BusinessException("图片来源不能为空，请提供 fileId 或 imageUrl");
        }

        SysPredLog predLog = new SysPredLog();
        predLog.setAlgorithmId(form.getAlgorithmId());
        if (form.getFileId() != null) {
            predLog.setOriginFileId(form.getFileId());
        }
        predLog.setOriginUrl(imageUrl);
        predLog.setStatus("processing");
        this.save(predLog);

        asyncTask.execute(predLog.getId(), form.getAlgorithmId(), imageUrl, form.getParams());

        PredictionResultVO vo = new PredictionResultVO();
        vo.setLogId(predLog.getId());
        vo.setStatus("processing");
        return vo;
    }

    @Override
    public Page<PredLogVO> getPredLogPage(PredLogQuery query) {
        Page<SysPredLog> page = new Page<>(query.getPageNum(), query.getPageSize());
        LambdaQueryWrapper<SysPredLog> wrapper = new LambdaQueryWrapper<SysPredLog>()
                .eq(query.getAlgorithmId() != null, SysPredLog::getAlgorithmId, query.getAlgorithmId())
                .orderByDesc(SysPredLog::getCreateTime);

        Page<SysPredLog> result = this.page(page, wrapper);
        Page<PredLogVO> voPage = new Page<>(result.getCurrent(), result.getSize(), result.getTotal());

        Set<Long> algorithmIds = result.getRecords().stream()
                .map(SysPredLog::getAlgorithmId)
                .filter(Objects::nonNull)
                .collect(Collectors.toSet());
        Map<Long, String> algorithmNameMap = algorithmIds.isEmpty()
                ? Collections.emptyMap()
                : algorithmService.listByIds(algorithmIds).stream()
                        .collect(Collectors.toMap(SysAlgorithm::getId, SysAlgorithm::getName, (a, b) -> a));

        voPage.setRecords(result.getRecords().stream().map(log -> {
            PredLogVO vo = new PredLogVO();
            BeanUtil.copyProperties(log, vo);
            vo.setAlgorithmName(algorithmNameMap.getOrDefault(log.getAlgorithmId(), "未知算法"));
            return vo;
        }).toList());
        return voPage;
    }

    private String resolveImageUrl(PredictionForm form) {
        if (form.getFileId() != null) {
            com.pei.dehaze.model.entity.SysFile sysFile = sysFileService.getById(form.getFileId());
            if (sysFile != null && CharSequenceUtil.isNotBlank(sysFile.getUrl())) {
                return toAbsoluteUrl(sysFile.getUrl());
            }
            log.warn("文件不存在或 URL 为空: fileId={}", form.getFileId());
            return null;
        }
        return toAbsoluteUrl(form.getImageUrl());
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
