package com.pei.dehaze.service.impl;

import cn.hutool.core.bean.BeanUtil;
import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.enums.LogStatusEnum;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysMemberMapper;
import com.pei.dehaze.mapper.SysPredLogMapper;
import com.pei.dehaze.model.entity.SysAlgorithm;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.model.entity.SysMember;
import com.pei.dehaze.model.entity.SysMemberBenefit;
import com.pei.dehaze.model.entity.SysPredLog;
import com.pei.dehaze.model.form.BatchPredictionForm;
import com.pei.dehaze.model.form.PredictionForm;
import com.pei.dehaze.model.query.PredLogQuery;
import com.pei.dehaze.model.vo.BatchPredictionResultVO;
import com.pei.dehaze.model.vo.PredLogVO;
import com.pei.dehaze.model.vo.PredictionQuotaVO;
import com.pei.dehaze.model.vo.PredictionResultVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.MemberBenefitService;
import com.pei.dehaze.service.SysAlgorithmService;
import com.pei.dehaze.service.SysFileService;
import com.pei.dehaze.service.SysPredLogService;
import com.pei.dehaze.service.impl.file.StorageServiceFactory;
import com.pei.dehaze.service.prediction.InterceptedResult;
import com.pei.dehaze.service.prediction.PredictionContext;
import com.pei.dehaze.service.prediction.PredictionInterceptorChain;
import io.micrometer.core.instrument.MeterRegistry;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.time.YearMonth;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class SysPredLogServiceImpl extends ServiceImpl<SysPredLogMapper, SysPredLog> implements SysPredLogService {

    private final SysAlgorithmService algorithmService;
    private final SysFileService sysFileService;
    private final StorageServiceFactory storageServiceFactory;
    private final PredLogAsyncTask asyncTask;
    private final PredictionInterceptorChain interceptorChain;
    private final MeterRegistry meterRegistry;
    private final SysMemberMapper memberMapper;
    private final MemberBenefitService memberBenefitService;

    @Override
    public PredictionResultVO predict(PredictionForm form) {
        long startTimeMs = System.currentTimeMillis();

        SysAlgorithm algorithm = algorithmService.getById(form.getAlgorithmId());
        if (algorithm == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "算法不存在");
        }

        SysFile originFile = form.getFileId() != null ? sysFileService.getById(form.getFileId()) : null;
        String imageUrl = resolveImageUrl(form, originFile);
        if (CharSequenceUtil.isBlank(imageUrl)) {
            throw new BusinessException("图片来源不能为空，请提供 fileId 或 imageUrl");
        }

        PredictionContext context = PredictionContext.builder()
                .algorithm(algorithm)
                .fileId(form.getFileId())
                .imageUrl(imageUrl)
                .originFile(originFile)
                .params(form.getParams())
                .startTimeMs(startTimeMs)
                .build();

        Optional<InterceptedResult> hit = interceptorChain.intercept(context);
        if (hit.isPresent()) {
            return buildCompletedResult(form, algorithm, originFile, imageUrl, hit.get(), startTimeMs);
        }

        SysPredLog predLog = new SysPredLog();
        predLog.setAlgorithmId(form.getAlgorithmId());
        if (form.getFileId() != null) {
            predLog.setOriginFileId(form.getFileId());
        }
        if (originFile != null) {
            predLog.setOriginMd5(originFile.getMd5());
        }
        predLog.setOriginUrl(imageUrl);
        predLog.setStatus(LogStatusEnum.PROCESSING);
        this.save(predLog);

        asyncTask.execute(predLog.getId(), form.getAlgorithmId(), imageUrl, form.getParams());

        PredictionResultVO vo = new PredictionResultVO();
        vo.setLogId(predLog.getId());
        vo.setStatus(LogStatusEnum.PROCESSING);
        return vo;
    }

    private PredictionResultVO buildCompletedResult(PredictionForm form, SysAlgorithm algorithm, SysFile originFile,
                                                    String imageUrl, InterceptedResult result, long startTimeMs) {
        int elapsed = (int) (System.currentTimeMillis() - startTimeMs);

        SysPredLog predLog = new SysPredLog();
        predLog.setAlgorithmId(algorithm.getId());
        if (form.getFileId() != null) {
            predLog.setOriginFileId(form.getFileId());
        }
        if (originFile != null) {
            predLog.setOriginMd5(originFile.getMd5());
        }
        predLog.setOriginUrl(imageUrl);
        predLog.setPredFileId(result.getResultFileId());
        predLog.setPredMd5(result.getResultMd5());
        predLog.setPredUrl(result.getResultUrl());
        predLog.setStatus(LogStatusEnum.COMPLETED);
        predLog.setTime(elapsed);
        this.save(predLog);

        meterRegistry.counter("dehaze_prediction_total", "status", "success").increment();

        PredictionResultVO vo = new PredictionResultVO();
        vo.setLogId(predLog.getId());
        vo.setStatus(LogStatusEnum.COMPLETED);
        vo.setResultUrl(result.getResultUrl());
        vo.setTime(elapsed);
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

    @Override
    public BatchPredictionResultVO batchPredict(BatchPredictionForm form) {
        Long userId = SecurityUtils.getUserId();
        SysMember member = memberMapper.selectOne(
                new LambdaQueryWrapper<SysMember>().eq(SysMember::getUserId, userId));
        int batchLimit = getBatchLimit(member);
        if (form.getItems().size() > batchLimit) {
            throw new BusinessException(ResultCode.BUSINESS_ERROR,
                    "批量处理图片数量不能超过" + batchLimit + "张");
        }

        List<PredictionResultVO> results = new ArrayList<>();
        for (BatchPredictionForm.BatchItem item : form.getItems()) {
            PredictionForm single = new PredictionForm();
            single.setAlgorithmId(form.getAlgorithmId());
            single.setFileId(item.getFileId());
            single.setImageUrl(item.getImageUrl());
            single.setParams(item.getParams());
            results.add(this.predict(single));
        }

        return BatchPredictionResultVO.builder()
                .total(results.size())
                .results(results)
                .build();
    }

    @Override
    public PredictionQuotaVO getQuota() {
        Long userId = SecurityUtils.getUserId();
        SysMember member = memberMapper.selectOne(
                new LambdaQueryWrapper<SysMember>().eq(SysMember::getUserId, userId));

        int total = 20;
        if (member != null) {
            SysMemberBenefit benefit = memberBenefitService.getByLevelCode(member.getLevelCode());
            if (benefit != null && benefit.getMonthlyDehazeQuota() != null) {
                total = benefit.getMonthlyDehazeQuota();
            } else if (member.getMonthlyDehazeQuota() != null) {
                total = member.getMonthlyDehazeQuota();
            }
        }

        int used = 0;
        if (member != null && member.getMonthlyDehazeUsed() != null) {
            used = member.getMonthlyDehazeUsed();
        } else {
            YearMonth currentMonth = YearMonth.now();
            LocalDate start = currentMonth.atDay(1);
            LocalDate end = currentMonth.atEndOfMonth();
            used = (int) this.count(new LambdaQueryWrapper<SysPredLog>()
                    .between(SysPredLog::getCreateTime, start.atStartOfDay(), end.atTime(23, 59, 59)));
        }

        int remaining = Math.max(0, total - used);
        LocalDate nextMonth = YearMonth.now().plusMonths(1).atDay(1);

        return PredictionQuotaVO.builder()
                .remaining(remaining)
                .total(total)
                .used(used)
                .resetDate(nextMonth.toString())
                .build();
    }

    private int getBatchLimit(SysMember member) {
        if (member != null && !"level_0".equals(member.getLevelCode())) {
            SysMemberBenefit benefit = memberBenefitService.getByLevelCode(member.getLevelCode());
            if (benefit != null && benefit.getBatchLimit() != null) {
                return benefit.getBatchLimit();
            }
        }
        return 5;
    }

    private String resolveImageUrl(PredictionForm form, SysFile originFile) {
        if (form.getFileId() != null) {
            if (originFile != null && originFile.getObjectName() != null && originFile.getStorage() != null) {
                return storageServiceFactory.get(originFile.getStorage()).getUrl(originFile.getObjectName());
            }
            log.warn("文件不存在或 objectName/storage 为空: fileId={}", form.getFileId());
            return null;
        }
        return form.getImageUrl();
    }
}
