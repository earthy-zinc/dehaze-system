package com.pei.dehaze.service.impl;

import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.enums.LogStatusEnum;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysAiModelMapper;
import com.pei.dehaze.mapper.SysAiProviderMapper;
import com.pei.dehaze.mapper.SysPredLogMapper;
import com.pei.dehaze.model.entity.SysAiModel;
import com.pei.dehaze.model.entity.SysAiProvider;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.model.entity.SysPredLog;
import com.pei.dehaze.model.form.PredictionForm;
import com.pei.dehaze.model.query.PredLogQuery;
import com.pei.dehaze.model.vo.PredLogVO;
import com.pei.dehaze.model.vo.PredictionResultVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.SysDictService;
import com.pei.dehaze.service.SysFileService;
import com.pei.dehaze.service.SysPredLogService;
import com.pei.dehaze.service.impl.file.StorageServiceFactory;
import com.pei.dehaze.service.prediction.InterceptedResult;
import com.pei.dehaze.service.prediction.PredictionContext;
import com.pei.dehaze.service.prediction.PredictionInterceptorChain;
import io.micrometer.core.instrument.MeterRegistry;
import java.util.List;
import java.util.Optional;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

/**
 * 图像任务编排服务：模型路由解析 → 拦截器链短路 → 落库 processing → 异步调用 Python 推理。
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class SysPredLogServiceImpl extends ServiceImpl<SysPredLogMapper, SysPredLog> implements SysPredLogService {

    private static final String IMAGE_MODEL_TYPE = "image_edit";
    private static final String DEFAULT_MODEL_DICT_TYPE = "ai_model";
    private static final String DEFAULT_IMAGE_MODEL_DICT_KEY = "default_model.image_edit";

    private final SysFileService sysFileService;
    private final StorageServiceFactory storageServiceFactory;
    private final PredLogAsyncTask asyncTask;
    private final PredictionInterceptorChain interceptorChain;
    private final MeterRegistry meterRegistry;
    private final SysAiModelMapper aiModelMapper;
    private final SysAiProviderMapper aiProviderMapper;
    private final SysDictService dictService;

    @Override
    public PredictionResultVO predict(PredictionForm form) {
        long startTimeMs = System.currentTimeMillis();

        SysAiModel model = resolveImageModel(form.getModel());
        String modelId = model.getModelId();

        SysFile originFile = form.getFileId() != null ? sysFileService.getById(form.getFileId()) : null;
        String imageUrl = resolveImageUrl(form, originFile);
        if (CharSequenceUtil.isBlank(imageUrl)) {
            throw new BusinessException(ResultCode.PARAM_IS_NULL, "图片来源不能为空，请提供 fileId 或 imageUrl");
        }

        PredictionContext context = PredictionContext.builder()
                .modelId(modelId)
                .fileId(form.getFileId())
                .imageUrl(imageUrl)
                .originFile(originFile)
                .params(form.getParams())
                .startTimeMs(startTimeMs)
                .build();

        Optional<InterceptedResult> hit = interceptorChain.intercept(context);
        if (hit.isPresent()) {
            return buildCompletedResult(form, model, originFile, imageUrl, hit.get(), startTimeMs);
        }

        SysPredLog predLog = new SysPredLog();
        predLog.setModelId(modelId);
        predLog.setProviderId(model.getProviderId());
        predLog.setActualModel(modelId);
        if (form.getFileId() != null) {
            predLog.setOriginFileId(form.getFileId());
        }
        if (originFile != null) {
            predLog.setOriginMd5(originFile.getMd5());
        }
        predLog.setOriginUrl(imageUrl);
        predLog.setRecommendedBy(form.getRecommendedBy());
        predLog.setStatus(LogStatusEnum.PROCESSING);
        this.save(predLog);

        asyncTask.execute(predLog.getId(), modelId, form.getFileId(), imageUrl, form.getParams());

        PredictionResultVO vo = new PredictionResultVO();
        vo.setTaskId(predLog.getId());
        vo.setStatus(LogStatusEnum.PROCESSING.getValue());
        return vo;
    }

    private PredictionResultVO buildCompletedResult(
            PredictionForm form,
            SysAiModel model,
            SysFile originFile,
            String imageUrl,
            InterceptedResult result,
            long startTimeMs) {
        int elapsedSec = (int) ((System.currentTimeMillis() - startTimeMs) / 1000);

        SysPredLog predLog = new SysPredLog();
        predLog.setModelId(model.getModelId());
        predLog.setProviderId(model.getProviderId());
        predLog.setActualModel(model.getModelId());
        if (form.getFileId() != null) {
            predLog.setOriginFileId(form.getFileId());
        }
        if (originFile != null) {
            predLog.setOriginMd5(originFile.getMd5());
        }
        predLog.setOriginUrl(imageUrl);
        predLog.setRecommendedBy(form.getRecommendedBy());
        predLog.setPredFileId(result.getResultFileId());
        predLog.setPredMd5(result.getResultMd5());
        predLog.setPredUrl(result.getResultUrl());
        predLog.setStatus(LogStatusEnum.COMPLETED);
        predLog.setTime(elapsedSec);
        this.save(predLog);

        meterRegistry.counter("dehaze_prediction_total", "status", "success").increment();

        PredictionResultVO vo = new PredictionResultVO();
        vo.setTaskId(predLog.getId());
        vo.setStatus(LogStatusEnum.COMPLETED.getValue());
        vo.setResultUrl(result.getResultUrl());
        vo.setResultMd5(result.getResultMd5());
        vo.setOriginMd5(originFile != null ? originFile.getMd5() : null);
        vo.setTime(elapsedSec);
        return vo;
    }

    @Override
    public Page<PredLogVO> getPredLogPage(PredLogQuery query) {
        Page<SysPredLog> page = new Page<>(query.getPageNum(), query.getPageSize());
        LambdaQueryWrapper<SysPredLog> wrapper = new LambdaQueryWrapper<SysPredLog>()
                .eq(CharSequenceUtil.isNotBlank(query.getModel()), SysPredLog::getModelId, query.getModel())
                // 归属过滤：仅返回当前用户的图像任务
                .eq(SysPredLog::getCreateBy, SecurityUtils.getUserId())
                .orderByDesc(SysPredLog::getCreateTime);

        Page<SysPredLog> result = this.page(page, wrapper);
        Page<PredLogVO> voPage = new Page<>(result.getCurrent(), result.getSize(), result.getTotal());
        voPage.setRecords(
                result.getRecords().stream().map(SysPredLogServiceImpl::toVo).toList());
        return voPage;
    }

    private static PredLogVO toVo(SysPredLog log) {
        PredLogVO vo = new PredLogVO();
        vo.setId(log.getId());
        vo.setModel(log.getModelId());
        vo.setOriginMd5(log.getOriginMd5());
        vo.setOriginUrl(log.getOriginUrl());
        vo.setPredMd5(log.getPredMd5());
        vo.setPredUrl(log.getPredUrl());
        vo.setStatus(log.getStatus());
        vo.setErrorMessage(log.getErrorMessage());
        vo.setTime(log.getTime());
        vo.setCreateTime(log.getCreateTime());
        return vo;
    }

    /**
     * 解析图像模型：请求显式 model 优先，其次平台默认（字典 ai_model.default_model.image_edit）；
     * 模型须存在、启用、类型为 image_edit 且其供应商可用，否则拒绝提交（对齐 Python resolve_image_model）。
     */
    private SysAiModel resolveImageModel(String model) {
        String requested = CharSequenceUtil.trim(model);
        String modelId = CharSequenceUtil.isBlank(requested)
                ? dictService.getStringValue(DEFAULT_MODEL_DICT_TYPE, DEFAULT_IMAGE_MODEL_DICT_KEY, null)
                : requested;
        if (CharSequenceUtil.isBlank(modelId)) {
            throw new BusinessException(
                    ResultCode.BUSINESS_ERROR, "平台未配置默认图像模型（字典 ai_model.default_model.image_edit 缺项）");
        }
        List<SysAiModel> rows =
                aiModelMapper.selectList(new LambdaQueryWrapper<SysAiModel>().eq(SysAiModel::getModelId, modelId));
        Optional<SysAiModel> matched = rows.stream()
                .filter(row ->
                        Integer.valueOf(1).equals(row.getStatus()) && IMAGE_MODEL_TYPE.equals(row.getModelType()))
                .findFirst();
        if (matched.isEmpty()) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "图像模型不可用: " + modelId);
        }
        SysAiModel available = matched.get();

        SysAiProvider provider = aiProviderMapper.selectById(available.getProviderId());
        if (provider == null || !Integer.valueOf(1).equals(provider.getStatus())) {
            throw new BusinessException(ResultCode.BUSINESS_ERROR, "模型 " + modelId + " 的供应商不可用");
        }
        return available;
    }

    private String resolveImageUrl(PredictionForm form, SysFile originFile) {
        if (form.getFileId() != null) {
            if (originFile == null) {
                throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "文件不存在: " + form.getFileId());
            }
            if (originFile.getObjectName() != null && originFile.getStorage() != null) {
                return storageServiceFactory.get(originFile.getStorage()).getUrl(originFile.getObjectName());
            }
            log.warn("文件不存在或 objectName/storage 为空: fileId={}", form.getFileId());
            return null;
        }
        return form.getImageUrl();
    }
}
