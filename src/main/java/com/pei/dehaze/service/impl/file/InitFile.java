package com.pei.dehaze.service.impl.file;

import cn.hutool.core.io.FileUtil;
import cn.hutool.core.io.file.PathUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.enums.ImageTypeEnum;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.util.FileUploadUtils;
import com.pei.dehaze.model.bo.DatasetItemBO;
import com.pei.dehaze.model.bo.FileBO;
import com.pei.dehaze.model.bo.PairedImage;
import com.pei.dehaze.model.entity.SysDataset;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.model.entity.SysWpxFile;
import com.pei.dehaze.service.*;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.Resource;
import org.jetbrains.annotations.NotNull;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Service
public class InitFile {
    @Value("${file.baseUrl}")
    private String baseUrl;

    @Value("${file.datasetPath}")
    private String datasetPath;

    @Value("${file.init}")
    private boolean init;

    @Resource
    private SysDatasetService sysDatasetService;

    @Resource
    private SysDatasetItemService sysDatasetItemService;

    @Resource
    private SysItemFileService sysItemFileService;

    @Resource
    private SysWpxFileService sysWpxFileService;

    @Resource
    private SysFileService sysFileService;

    /**
     * 文件保存后下次仅需创建 datasetItem 和 itemFile 映射关系表即可 仅需删除这两张表
     */
    @PostConstruct
    public void initDataset() {
        if (!init) return;
        // 查询数据库，获取叶子节点数据集id
        List<Long> datasetIds = sysDatasetService.getLeafDatasetIds();

        for (Long datasetId : datasetIds) {
            // 获取当前数据集的所有数据项，整理为列表
            ArrayList<PairedImage> pairedImages = getPairedImages(datasetId);
            // 针对每一个数据项，进行上传
            pairedImages.forEach(pairedImage -> {
                SysDatasetItem datasetItem = sysDatasetItemService.createDatasetItem(datasetId);
                Long itemId = datasetItem.getId();

                String cleanPath = pairedImage.getCleanPath();
                DatasetItemBO cleanBO = createDatasetItemBO(cleanPath, ImageTypeEnum.CLEAN);
                sysItemFileService.saveItemFile(itemId, cleanBO);

                List<String> hazePath = pairedImage.getHazePath();
                hazePath.forEach(haze -> {
                    DatasetItemBO hazeBO = createDatasetItemBO(haze, ImageTypeEnum.HAZE);
                    sysItemFileService.saveItemFile(itemId, hazeBO);
                });
            });
        }
        initWpxFile();
    }

    private ArrayList<PairedImage> getPairedImages(long id) {
        // 获取数据集信息并验证类型
        SysDataset sysDataset = sysDatasetService.getById(id);
        String datasetType = sysDataset.getType();
        if (!"图像去雾".equals(datasetType)) {
            throw new BusinessException("暂不支持非去雾数据集初始化");
        }

        // 获取数据集路径和文件夹标识
        String filePath = sysDataset.getPath();
        Path datasetBasePath = Path.of(datasetPath, filePath);
        List<String> hazeFlags = Arrays.asList("haze", "hazy");
        List<String> cleanFlags = Arrays.asList("clean", "clear", "gt", "GT");

        // 获取雾霾图像和清晰图像文件夹路径
        String hazeFlag = getValidPath(hazeFlags, datasetBasePath);
        String cleanFlag = getValidPath(cleanFlags, datasetBasePath);

        if (hazeFlag == null || cleanFlag == null) {
            throw new BusinessException("数据集目录" + filePath + "下未找到清晰图像或雾霾图像文件夹");
        }

        Path hazePath = datasetBasePath.resolve(hazeFlag);
        Path cleanPath = datasetBasePath.resolve(cleanFlag);
        if (!PathUtil.isDirectory(hazePath) || !PathUtil.isDirectory(cleanPath)) {
            throw new BusinessException("数据集目录" + filePath + "下未找到清晰图像或雾霾图像文件夹");
        }

        // 获取并处理图像文件列表
        List<String> hazeImages = getSortedDistinctFileNames(hazePath);
        List<String> cleanImages = getSortedDistinctFileNames(cleanPath);

        // 校验成对图片数量
        return getPairImages(hazeImages, cleanImages);
    }

    private DatasetItemBO createDatasetItemBO(String filePath, ImageTypeEnum type) {
        File file = new File(filePath);
        if (!file.exists() && !file.isFile()) {
            throw new BusinessException("File not found: " + filePath);
        }
        // 获取文件所在文件夹路径
        Path fileDirPath = Paths.get(filePath).getParent();
        Path dataset = Paths.get(datasetPath);
        // 文件所在文件夹相对路径
        String dirRelPath = dataset.relativize(fileDirPath).toString().replace("\\", "/");
        String fileName = file.getName();

        try (FileInputStream stream = new FileInputStream(file)) {
            String md5 = FileUploadUtils.getMd5(stream);
            String suffix = FileUtil.getSuffix(fileName);
            String objectName = dirRelPath + "/" + md5 + "." + suffix;
            String url = baseUrl + "/" + objectName;

            DatasetItemBO itemBO = new DatasetItemBO();
            itemBO.setFile(file);
            itemBO.setName(fileName);
            itemBO.setObjectName(objectName);
            itemBO.setExtension(suffix);
            itemBO.setMd5(md5);
            itemBO.setPath(objectName);
            itemBO.setSize(file.length());
            itemBO.setUrl(url);
            itemBO.setType(type.getValue());
            return itemBO;
        } catch (IOException e) {
            throw new BusinessException("转换BO失败");
        }
    }

    private void initWpxFile() {
        List<SysWpxFile> sysWpxFiles = sysWpxFileService.list();
        for (SysWpxFile sysWpxFile : sysWpxFiles) {
            // 根据原始图片md5查询原始文件id
            String originMd5 = sysWpxFile.getOriginMd5();
            SysFile originSysFile = sysFileService.getOne(new LambdaQueryWrapper<SysFile>().eq(SysFile::getMd5, originMd5));
            sysWpxFile.setOriginFileId(originSysFile.getId());

            // 根据新文件路径上传新图片
            String newPath = sysWpxFile.getNewPath();
            Path path = Paths.get(datasetPath, newPath);
            FileBO fileBO = FileUploadUtils.createFileBO(new File(path.toAbsolutePath().toString()), baseUrl, newPath);
            SysFile newFile = sysFileService.saveFile(fileBO);
            sysWpxFile.setNewFileId(newFile.getId());
            sysWpxFileService.updateById(sysWpxFile);
        }
    }

    private static String getValidPath(List<String> flags, Path basePath) {
        for (String flag : flags) {
            Path path = basePath.resolve(flag);
            if (PathUtil.isDirectory(path)) {
                return flag;
            }
        }
        return null;
    }

    // 获取文件夹下排序去重后的文件名列表
    private static List<String> getSortedDistinctFileNames(Path folderPath) {
        String path = folderPath.toAbsolutePath().toString();
        List<String> fileNames = FileUtil.listFileNames(path);
        return fileNames
                .stream()
                .map(filename -> Path.of(path, filename).toString())
                .sorted()
                .distinct()
                .toList();
    }

    @NotNull
    private static ArrayList<PairedImage> getPairImages(List<String> hazeImages, List<String> cleanImages) {
        if (hazeImages.size() % cleanImages.size() != 0) {
            throw new BusinessException("成对图片数量不符");
        }
        int hazeCount = hazeImages.size() / cleanImages.size();

        // 构建成对的图像列表
        ArrayList<PairedImage> pairedImageList = new ArrayList<>();
        for (int i = 0; i < cleanImages.size(); i++) {
            List<String> haze = new ArrayList<>();
            String clean = cleanImages.get(i); // 使用索引访问，避免修改集合
            for (int j = 0; j < hazeCount; j++) {
                haze.add(hazeImages.get(i * hazeCount + j)); // 获取对应的雾霾图像
            }

            PairedImage pairedImage = new PairedImage();
            pairedImage.setCleanPath(clean);
            pairedImage.setHazePath(haze);
            pairedImageList.add(pairedImage);
        }
        return pairedImageList;
    }
}
