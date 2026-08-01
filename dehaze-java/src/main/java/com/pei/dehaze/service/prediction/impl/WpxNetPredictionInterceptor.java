package com.pei.dehaze.service.prediction.impl;

import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.model.entity.SysAlgorithm;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.model.entity.SysWpxFile;
import com.pei.dehaze.service.SysAlgorithmService;
import com.pei.dehaze.service.SysFileService;
import com.pei.dehaze.service.SysWpxFileService;
import com.pei.dehaze.service.impl.file.StorageServiceFactory;
import com.pei.dehaze.service.prediction.InterceptedResult;
import com.pei.dehaze.service.prediction.PredictionContext;
import com.pei.dehaze.service.prediction.PredictionInterceptor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.Optional;

/**
 * WPXNet 系列算法预查询拦截器。
 * <p>
 * 当算法为 WPXNet 子算法且原始图 MD5 在 sys_wpx_file 表中存在映射时，
 * 直接返回已处理好的去雾图，跳过 Python 算法调用。
 * <p>
 * 前置条件：sys_wpx_file 表已由 scripts/init_wpx_file.py 写入映射数据。
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class WpxNetPredictionInterceptor implements PredictionInterceptor {

    private static final String WPXNET_ROOT_NAME = "WPXNet";

    private final SysAlgorithmService algorithmService;
    private final SysWpxFileService wpxFileService;
    private final SysFileService fileService;
    private final StorageServiceFactory storageServiceFactory;

    @Override
    public Optional<InterceptedResult> intercept(PredictionContext context) {
        SysAlgorithm algorithm = context.getAlgorithm();

        SysAlgorithm root = algorithmService.getRootAlgorithm(algorithm.getId());
        if (!root.getName().contains(WPXNET_ROOT_NAME)) {
            return Optional.empty();
        }

        SysFile originFile = context.getOriginFile();
        if (originFile == null || CharSequenceUtil.isBlank(originFile.getMd5())) {
            return Optional.empty();
        }
        String originMd5 = originFile.getMd5();

        SysWpxFile wpxFile = wpxFileService.getOne(
                new LambdaQueryWrapper<SysWpxFile>().eq(SysWpxFile::getOriginMd5, originMd5));
        if (wpxFile == null || wpxFile.getNewFileId() == null) {
            log.debug("WPXNet 命中算法但未找到映射: algorithmId={}, originMd5={}", algorithm.getId(), originMd5);
            return Optional.empty();
        }

        SysFile newFile = fileService.getById(wpxFile.getNewFileId());
        if (newFile == null) {
            log.warn("WPXNet 映射的 newFileId 不存在: wpxFileId={}, newFileId={}", wpxFile.getId(), wpxFile.getNewFileId());
            return Optional.empty();
        }

        String resultUrl = storageServiceFactory.get(newFile.getStorage()).getUrl(newFile.getObjectName());
        log.debug("WPXNet 预查询命中: algorithmId={}, originMd5={}, resultUrl={}",
                algorithm.getId(), originMd5, resultUrl);

        return Optional.of(InterceptedResult.builder()
                .resultUrl(resultUrl)
                .resultMd5(newFile.getMd5())
                .resultFileId(newFile.getId())
                .build());
    }
}
