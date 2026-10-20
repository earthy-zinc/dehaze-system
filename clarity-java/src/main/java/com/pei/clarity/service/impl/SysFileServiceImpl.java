package com.pei.clarity.service.impl;

import cn.hutool.core.io.FileUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.clarity.common.exception.BusinessException;
import com.pei.clarity.common.result.ResultCode;
import com.pei.clarity.mapper.SysFileMapper;
import com.pei.clarity.model.dto.FileDTO;
import com.pei.clarity.model.entity.SysFile;
import com.pei.clarity.security.util.SecurityUtils;
import com.pei.clarity.service.FileService;
import com.pei.clarity.service.SysFileService;
import com.pei.clarity.service.impl.file.StorageServiceFactory;
import io.micrometer.core.instrument.MeterRegistry;
import java.io.InputStream;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

/**
 * @author earthy-zinc
 * @since 2024-06-08 18:38:14
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class SysFileServiceImpl extends ServiceImpl<SysFileMapper, SysFile> implements SysFileService {

    private final StorageServiceFactory storageServiceFactory;
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
                convertSize(fileDTO.getSize()),
                fileDTO.getSize(),
                userId);
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
                convertSize(fileDTO.getSize()),
                fileDTO.getSize(),
                userId);
        return this.getOne(new LambdaQueryWrapper<SysFile>()
                .eq(SysFile::getMd5, fileDTO.getMd5())
                .last("LIMIT 1"));
    }

    /**
     * 文件大小人可读格式化（python convert_size 同款口径：0 → "0B"）。
     * hutool readableFileSize 对 0 返回 "0"，三端共享 sys_file.size 列需口径一致。
     */
    private String convertSize(long size) {
        if (size == 0) {
            return "0B";
        }
        return FileUtil.readableFileSize(size);
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
