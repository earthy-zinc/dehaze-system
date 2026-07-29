package com.pei.dehaze.service.impl;

import cn.hutool.core.io.FileUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysFileMapper;
import com.pei.dehaze.model.bo.FileBO;
import com.pei.dehaze.model.entity.SysAlgorithm;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.model.entity.SysWpxFile;
import com.pei.dehaze.service.FileService;
import com.pei.dehaze.service.SysAlgorithmService;
import com.pei.dehaze.service.SysFileService;
import com.pei.dehaze.service.SysWpxFileService;
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

    private final FileService fileService;
    private final SysWpxFileService sysWpxFileService;
    private final SysAlgorithmService sysAlgorithmService;

    @Override
    public SysFile check(String md5) {
        return this.getOne(new LambdaQueryWrapper<SysFile>().eq(SysFile::getMd5, md5));
    }

    @Override
    public SysFile saveFile(FileBO fileBO) {
        SysFile sysFile = baseMapper.selectByMd5IncludeDeleted(fileBO.getMd5());
        if (sysFile != null) {
            if (sysFile.getDeleted() != null && sysFile.getDeleted() == 1) {
                baseMapper.hardDeleteById(sysFile.getId());
            } else {
                return sysFile;
            }
        }

        fileBO = fileService.uploadFile(fileBO);
        sysFile = SysFile.builder()
                .name(fileBO.getName())
                .objectName(fileBO.getObjectName())
                .size(FileUtil.readableFileSize(fileBO.getSize()))
                .type(fileBO.getExtension())
                .url(fileBO.getUrl())
                .md5(fileBO.getMd5())
                .path(fileBO.getPath())
                .build();
        this.save(sysFile);
        return sysFile;
    }

    @Override
    public SysFile saveFileRecord(FileBO fileBO) {
        SysFile sysFile = baseMapper.selectByMd5IncludeDeleted(fileBO.getMd5());
        if (sysFile != null) {
            if (sysFile.getDeleted() != null && sysFile.getDeleted() == 1) {
                baseMapper.hardDeleteById(sysFile.getId());
            } else {
                return sysFile;
            }
        }

        sysFile = SysFile.builder()
                .name(fileBO.getName())
                .objectName(fileBO.getObjectName())
                .size(FileUtil.readableFileSize(fileBO.getSize()))
                .type(fileBO.getExtension())
                .url(fileBO.getUrl())
                .md5(fileBO.getMd5())
                .path(fileBO.getPath())
                .build();
        this.save(sysFile);
        return sysFile;
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
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "不存在当前文件");
        }
        boolean result = fileService.deleteFile(sysFile.getObjectName());
        if (!result) {
            throw new BusinessException("删除文件失败");
        }
        baseMapper.hardDeleteById(fileId);
        return true;
    }

    @Override
    public InputStream download(String objectName) {
        return fileService.downLoadFile(objectName);
    }
}
