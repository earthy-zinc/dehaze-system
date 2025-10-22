package com.pei.dehaze.service.impl;

import cn.hutool.core.lang.Assert;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.util.FileUploadUtils;
import com.pei.dehaze.common.util.ImageUtils;
import com.pei.dehaze.mapper.SysItemFileMapper;
import com.pei.dehaze.model.bo.DatasetItemBO;
import com.pei.dehaze.model.dto.ImageFileInfo;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.model.vo.ImageUrlVO;
import com.pei.dehaze.service.SysFileService;
import com.pei.dehaze.service.SysItemFileService;
import jakarta.annotation.Resource;
import org.jetbrains.annotations.NotNull;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

@Service
public class SysItemFileServiceImpl extends ServiceImpl<SysItemFileMapper, SysItemFile>
        implements SysItemFileService {

    @Resource
    private SysFileService sysFileService;

    @Override
    public ImageFileInfo saveItemFile(Long itemId, DatasetItemBO itemBO) {
        // 保存源文件
        SysFile sysFile = sysFileService.saveFile(itemBO);
        // 生成缩略图并保存
        DatasetItemBO thumbnailItemBO = getThumbnailItemBO(itemBO);
        SysFile thumbnailSysFile = sysFileService.saveFile(thumbnailItemBO);

        // 查询是否已经存在关联关系
        SysItemFile sysItemFile = this.getOne(new LambdaQueryWrapper<SysItemFile>()
                .eq(SysItemFile::getFileId, sysFile.getId())
                .eq(SysItemFile::getThumbnailFileId, thumbnailSysFile.getId()));

        if (sysItemFile == null) {
            // 保存数据项与文件关联关系
            sysItemFile = new SysItemFile();
            sysItemFile.setItemId(itemId);
            sysItemFile.setFileId(sysFile.getId());
            sysItemFile.setThumbnailFileId(thumbnailSysFile.getId());
            sysItemFile.setType(itemBO.getType());
            sysItemFile.setDescription(itemBO.getDescription());
            this.save(sysItemFile);
        }

        return ImageFileInfo.builder()
                .id(sysItemFile.getId())
                .datasetItemId(sysItemFile.getItemId())
                .fileId(sysItemFile.getFileId())
                .type(sysItemFile.getType())
                .description(sysItemFile.getDescription())
                .url(sysFile.getUrl())
                .build();
    }

    @Override
    public List<ImageUrlVO> getImageUrlVOs(Long itemId) {
        return this.baseMapper.listImageUrl(itemId);
    }

    @Override
    public boolean deleteItemFile(Long itemId) {
        SysItemFile sysItemFile = this.getById(itemId);
        Assert.notNull(sysItemFile, "未查询到对应数据项");

        Long fileId = sysItemFile.getFileId();
        Long thumbnailFileId = sysItemFile.getThumbnailFileId();

        boolean res1 = sysFileService.deleteFile(fileId);
        Assert.isTrue(res1, "删除原图失败");

        boolean res2 = sysFileService.deleteFile(thumbnailFileId);
        Assert.isTrue(res2, "删除缩略图失败");

        return this.removeById(itemId);
    }

    @NotNull
    private static DatasetItemBO getThumbnailItemBO(DatasetItemBO itemBO) {
        File thumbnailFile = ImageUtils.generateThumbnail(itemBO.getFile(), 400, 400);

        try (InputStream thumbnailInputStream = new FileInputStream(thumbnailFile)) {
            long size = thumbnailFile.length();
            String md5 = FileUploadUtils.getMd5(thumbnailInputStream);
            String extension = itemBO.getExtension();
            String name = addSuffix(itemBO.getName(), "_thumbnail");
            String originPath = Paths.get(itemBO.getPath()).getParent().toString();
            String objectName = Path.of("thumbnail", originPath, md5 + "." + extension).toString().replace("\\", "/");
            String path = Path.of("thumbnail", objectName).toString();

            DatasetItemBO thumbnailItemBO = new DatasetItemBO();
            thumbnailItemBO.setFile(thumbnailFile);
            thumbnailItemBO.setName(name);
            thumbnailItemBO.setObjectName(objectName);
            thumbnailItemBO.setExtension(extension);
            thumbnailItemBO.setMd5(md5);
            thumbnailItemBO.setPath(path);
            thumbnailItemBO.setSize(size);
            thumbnailItemBO.setDescription(itemBO.getDescription());
            thumbnailItemBO.setType(itemBO.getType());
            return thumbnailItemBO;
        } catch (Exception e) {
            throw new BusinessException("生成缩略图失败", e);
        }
    }

    private static String addSuffix(String fileName, String suffix) {
        // 找到文件名中最后一个"."的位置
        int dotIndex = fileName.lastIndexOf(".");

        // 如果文件名中没有"."，说明没有后缀，直接返回原文件名
        if (dotIndex == -1) {
            return fileName + suffix;
        }

        // 提取文件名部分和扩展名部分
        String namePart = fileName.substring(0, dotIndex); // 文件名
        String extensionPart = fileName.substring(dotIndex); // 文件扩展名（包含点）

        // 在文件名后添加"_thumbnail"并返回结果
        return namePart + suffix + extensionPart;
    }
}
