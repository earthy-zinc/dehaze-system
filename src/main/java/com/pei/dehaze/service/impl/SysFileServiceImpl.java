package com.pei.dehaze.service.impl;

import cn.hutool.core.io.FileUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.mapper.SysFileMapper;
import com.pei.dehaze.model.bo.FileBO;
import com.pei.dehaze.model.entity.SysAlgorithm;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.model.entity.SysWpxFile;
import com.pei.dehaze.service.FileService;
import com.pei.dehaze.service.SysAlgorithmService;
import com.pei.dehaze.service.SysFileService;
import com.pei.dehaze.service.SysWpxFileService;
import jakarta.annotation.Resource;
import org.springframework.stereotype.Service;

import java.io.InputStream;

/**
 * @author earthy-zinc
 * @since 2024-06-08 18:38:14
 */
@Service
public class SysFileServiceImpl extends ServiceImpl<SysFileMapper, SysFile> implements SysFileService {
    @Resource
    private FileService fileService;

    @Resource
    private SysWpxFileService sysWpxFileService;

    @Resource
    private SysAlgorithmService sysAlgorithmService;

    @Override
    public boolean check(String md5) {
        return false;
    }

    @Override
    public SysFile saveFile(FileBO fileBO) {
        // 先根据md5查询，如果存在则直接返回
        SysFile sysFile = this.getOne(new LambdaQueryWrapper<SysFile>().eq(SysFile::getMd5, fileBO.getMd5()));
        if (sysFile != null) return sysFile;

        // 如果不存在，则上传文件
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
    public SysFile getWpxFile(SysFile oldFile, Long modelId) {
        // 利用sysWpxFileService查询一条originMd5为fileInfo.getOriginMd5()的数据
        SysAlgorithm algorithm = sysAlgorithmService.getRootAlgorithm(modelId);
        if (!algorithm.getName().contains("WPXNet")) return oldFile;

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
            throw new BusinessException("不存在当前文件");
        }
        boolean result = fileService.deleteFile(sysFile.getObjectName());
        if (!result) {
            throw new BusinessException("删除文件失败");
        }
        return this.removeById(fileId);
    }

    @Override
    public InputStream download(String objectName) {
        return fileService.downLoadFile(objectName);
    }
}
