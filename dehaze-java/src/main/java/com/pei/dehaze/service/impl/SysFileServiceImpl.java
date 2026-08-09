package com.pei.dehaze.service.impl;

import cn.hutool.core.io.FileUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysFileMapper;
import com.pei.dehaze.model.dto.FileDTO;
import com.pei.dehaze.model.entity.SysAlgorithm;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.model.entity.SysWpxFile;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.FileService;
import com.pei.dehaze.service.SysAlgorithmService;
import com.pei.dehaze.service.SysFileService;
import com.pei.dehaze.service.SysWpxFileService;
import com.pei.dehaze.service.impl.file.StorageServiceFactory;
import io.micrometer.core.instrument.MeterRegistry;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.io.InputStream;

/**
 * @author earthy-zinc
 * @since 2024-06-08 18:38:14
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class SysFileServiceImpl extends ServiceImpl<SysFileMapper, SysFile> implements SysFileService {

    /**
     * WPXNet 算法名称关键字
     */
    private static final String WPXNET_ALGORITHM_NAME = "WPXNet";

    private final StorageServiceFactory storageServiceFactory;
    private final SysWpxFileService sysWpxFileService;
    private final SysAlgorithmService sysAlgorithmService;
    private final MeterRegistry meterRegistry;

    @Override
    public SysFile check(String md5) {
        return this.getOne(new LambdaQueryWrapper<SysFile>().eq(SysFile::getMd5, md5));
    }

    @Override
    public SysFile saveFile(FileDTO fileDTO) {
        fileDTO = storageServiceFactory.getDefault().uploadFile(fileDTO);
        Long userId = SecurityUtils.getUserId();
        baseMapper.upsertByMd5(
                fileDTO.getMd5(),
                fileDTO.getExtension(),
                fileDTO.getName(),
                fileDTO.getObjectName(),
                fileDTO.getStorage(),
                FileUtil.readableFileSize(fileDTO.getSize()),
                fileDTO.getSize(),
                userId
        );
        meterRegistry.counter("dehaze_file_upload_total").increment();
        return this.getOne(new LambdaQueryWrapper<SysFile>()
                .eq(SysFile::getMd5, fileDTO.getMd5())
                .last("LIMIT 1"));
    }

    @Override
    public SysFile saveFileRecord(FileDTO fileDTO) {
        Long userId = SecurityUtils.getUserId();
        baseMapper.upsertByMd5(
                fileDTO.getMd5(),
                fileDTO.getExtension(),
                fileDTO.getName(),
                fileDTO.getObjectName(),
                fileDTO.getStorage(),
                FileUtil.readableFileSize(fileDTO.getSize()),
                fileDTO.getSize(),
                userId
        );
        return this.getOne(new LambdaQueryWrapper<SysFile>()
                .eq(SysFile::getMd5, fileDTO.getMd5())
                .last("LIMIT 1"));
    }

    @Override
    public SysFile getWpxFile(SysFile oldFile, Long modelId) {
        // 利用sysWpxFileService查询一条originMd5为fileInfo.getOriginMd5()的数据
        SysAlgorithm algorithm = sysAlgorithmService.getRootAlgorithm(modelId);
        if (!algorithm.getName().contains(WPXNET_ALGORITHM_NAME)) return oldFile;

        LambdaQueryWrapper<SysWpxFile> queryWrapper = new LambdaQueryWrapper<SysWpxFile>().eq(SysWpxFile::getOriginMd5, oldFile.getMd5());
        SysWpxFile sysWpxFile = sysWpxFileService.getOne(queryWrapper);
        if (sysWpxFile == null) return oldFile;

        SysFile newFile = this.getOne(new LambdaQueryWrapper<SysFile>().eq(SysFile::getMd5, sysWpxFile.getNewMd5()));

        if (newFile == null) throw new BusinessException("无法从SysFile获取映射到的文件信息");
        return newFile;
    }

    @Override
    public boolean deleteFile(Long fileId) {
        SysFile sysFile = this.getById(fileId);
        if (sysFile == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "文件不存在");
        }
        // 先删DB元数据（事务内），再删物理文件（best-effort，失败仅记录日志，孤儿文件由定时任务清理）
        baseMapper.deleteById(fileId);
        FileService storage = storageServiceFactory.get(sysFile.getStorage());
        if (!storage.deleteFile(sysFile.getObjectName())) {
            log.warn("物理文件删除失败（孤儿文件待清理）, objectName: {}", sysFile.getObjectName());
        }
        return true;
    }

    @Override
    public InputStream download(String objectName) {
        SysFile sysFile = this.getOne(new LambdaQueryWrapper<SysFile>().eq(SysFile::getObjectName, objectName));
        if (sysFile == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "文件不存在: " + objectName);
        }
        return storageServiceFactory.get(sysFile.getStorage()).downLoadFile(objectName);
    }

    @Override
    public void fillUrl(SysFile file) {
        if (file != null && file.getStorage() != null && file.getObjectName() != null) {
            file.setUrl(storageServiceFactory.get(file.getStorage()).getUrl(file.getObjectName()));
        }
    }
}
