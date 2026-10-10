package com.pei.dehaze.service.prediction.impl;

import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.model.entity.SysWpxFile;
import com.pei.dehaze.service.SysFileService;
import com.pei.dehaze.service.SysWpxFileService;
import com.pei.dehaze.service.impl.file.StorageServiceFactory;
import com.pei.dehaze.service.prediction.InterceptedResult;
import com.pei.dehaze.service.prediction.PredictionContext;
import com.pei.dehaze.service.prediction.PredictionInterceptor;
import java.util.Optional;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * WPXNet 系列模型预查询拦截器。
 * <p>
 * 当图像模型为 WPXNet 系（model_id 前缀 {@value #WPXNET_MODEL_PREFIX}）且原始图 MD5 在 sys_wpx_file
 * 表中存在映射时，直接返回已处理好的去雾图，跳过 Python 推理调用。
 * <p>
 * 前置条件：sys_wpx_file 表已由 scripts/init_wpx_file.py 写入映射数据。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class WpxNetPredictionInterceptor implements PredictionInterceptor {

    private static final String WPXNET_MODEL_PREFIX = "wpxnet-";

    private final SysWpxFileService wpxFileService;
    private final SysFileService fileService;
    private final StorageServiceFactory storageServiceFactory;

    @Override
    public Optional<InterceptedResult> intercept(PredictionContext context) {
        if (context.getModelId() == null || !context.getModelId().startsWith(WPXNET_MODEL_PREFIX)) {
            return Optional.empty();
        }

        SysFile originFile = context.getOriginFile();
        if (originFile == null || CharSequenceUtil.isBlank(originFile.getMd5())) {
            return Optional.empty();
        }
        String originMd5 = originFile.getMd5();

        SysWpxFile wpxFile =
                wpxFileService.getOne(new LambdaQueryWrapper<SysWpxFile>().eq(SysWpxFile::getOriginMd5, originMd5));
        if (wpxFile == null || wpxFile.getNewFileId() == null) {
            log.debug("WPXNet 命中模型但未找到映射: modelId={}, originMd5={}", context.getModelId(), originMd5);
            return Optional.empty();
        }

        SysFile newFile = fileService.getById(wpxFile.getNewFileId());
        if (newFile == null) {
            log.warn("WPXNet 映射的 newFileId 不存在: wpxFileId={}, newFileId={}", wpxFile.getId(), wpxFile.getNewFileId());
            return Optional.empty();
        }

        String resultUrl = storageServiceFactory.get(newFile.getStorage()).getUrl(newFile.getObjectName());
        log.debug("WPXNet 预查询命中: modelId={}, originMd5={}, resultUrl={}", context.getModelId(), originMd5, resultUrl);

        return Optional.of(InterceptedResult.builder()
                .resultUrl(resultUrl)
                .resultMd5(newFile.getMd5())
                .resultFileId(newFile.getId())
                .build());
    }
}
