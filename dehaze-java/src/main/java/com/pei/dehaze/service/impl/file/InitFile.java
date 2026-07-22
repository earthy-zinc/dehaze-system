package com.pei.dehaze.service.impl.file;

import cn.hutool.core.io.FileUtil;
import cn.hutool.core.io.file.PathUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.dehaze.common.enums.ImageTypeEnum;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.util.FileUploadUtils;
import com.pei.dehaze.model.bo.ItemFileBO;
import com.pei.dehaze.model.bo.FileBO;
import com.pei.dehaze.model.bo.PairedImage;
import com.pei.dehaze.model.entity.SysDataset;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.model.entity.SysWpxFile;
import com.pei.dehaze.service.*;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * 数据集初始化：扫描磁盘目录，创建 datasetItem / itemFile / file 记录。
 * <p>
 * 源文件由 nginx-dataset (端口 9000) 直服，不上传 MinIO；
 * 缩略图生成后上传 MinIO（派生小文件，nginx 无此资源）。
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class InitFile {

    private static final String DATASET_TYPE_DEHAZE = "图像去雾";
    private static final List<String> HAZE_FOLDER_FLAGS = List.of("haze", "hazy");
    private static final List<String> CLEAN_FOLDER_FLAGS = List.of("clean", "clear", "gt");

    @Value("${file.baseUrl}")
    private String baseUrl;

    @Value("${file.datasetPath}")
    private String datasetPath;

    /**
     * nginx-dataset 静态服务基础 URL，如 http://127.0.0.1:9000
     * 数据集源文件由此服务直服，无需上传 MinIO
     */
    @Value("${file.datasetBaseUrl}")
    private String datasetBaseUrl;

    @Value("${file.init}")
    private boolean init;

    private final SysDatasetService sysDatasetService;
    private final SysDatasetItemService sysDatasetItemService;
    private final SysItemFileService sysItemFileService;
    private final SysWpxFileService sysWpxFileService;
    private final SysFileService sysFileService;

    /**
     * 文件保存后下次仅需创建 datasetItem 和 itemFile 映射关系表即可 仅需删除这两张表。
     * <p>
     * 通过 {@link ApplicationReadyEvent} 在 Spring 上下文完全就绪、Tomcat 已绑定 8989 端口后触发，
     * 避免在 {@code @PostConstruct} 阶段（上下文刷新期间）抢占数据库连接、阻塞端口监听。
     * <p>
     * 使用 {@code @Async("datasetTaskExecutor")} 复用既有线程池，
     * 自动获得 SecurityContext/MDC traceId 传播、统一异常处理与优雅关闭。
     */
    @EventListener(ApplicationReadyEvent.class)
    @Async("datasetTaskExecutor")
    public void initDataset() {
        if (!init) return;
        try {
            doInit();
        } catch (Exception e) {
            log.error("数据集初始化异常", e);
        }
    }

    private void doInit() {
        log.info("开始初始化数据集，datasetPath={}, datasetBaseUrl={}", datasetPath, datasetBaseUrl);

        // 查询数据库，获取叶子节点数据集id
        List<Long> datasetIds = sysDatasetService.getLeafDatasetIds();

        for (Long datasetId : datasetIds) {
            try {
                initSingleDataset(datasetId);
            } catch (Exception e) {
                log.warn("数据集[{}]初始化跳过: {}", datasetId, e.getMessage());
            }
        }
        initWpxFile();
        log.info("数据集初始化完成");
    }

    private void initSingleDataset(Long datasetId) {
        // 幂等：已有数据项的数据集跳过
        long existingCount = sysDatasetItemService.count(
                new com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<SysDatasetItem>()
                        .eq(SysDatasetItem::getDatasetId, datasetId));
        if (existingCount > 0) {
            log.info("数据集[{}]已有{}条数据项，跳过初始化", datasetId, existingCount);
            return;
        }

        // 获取当前数据集的所有数据项，整理为列表
        ArrayList<PairedImage> pairedImages = getPairedImages(datasetId);
        log.info("数据集[{}]发现{}组配对图片", datasetId, pairedImages.size());

        // 针对每一个数据项，创建记录
        for (PairedImage pairedImage : pairedImages) {
            SysDatasetItem datasetItem = sysDatasetItemService.createDatasetItem(datasetId, null);
            Long itemId = datasetItem.getId();

            // 清晰图：nginx 直服，不上传 MinIO
            String cleanPath = pairedImage.getCleanPath();
            saveSourceFileRecord(itemId, cleanPath, ImageTypeEnum.CLEAN);

            // 雾图：nginx 直服，不上传 MinIO
            List<String> hazePaths = pairedImage.getHazePath();
            for (String hazePath : hazePaths) {
                saveSourceFileRecord(itemId, hazePath, ImageTypeEnum.HAZE);
            }
        }
    }

    /**
     * 为源文件创建 DB 记录（不上传 MinIO，不生成缩略图）
     * 缩略图可后续通过批量任务生成
     */
    private void saveSourceFileRecord(Long itemId, String filePath, ImageTypeEnum type) {
        File file = new File(filePath);
        if (!file.exists() || !file.isFile()) {
            throw new BusinessException("File not found: " + filePath);
        }

        Path fileDirPath = Paths.get(filePath).getParent();
        Path dataset = Paths.get(datasetPath);
        // 文件相对路径（如 Dense-Haze/clean）
        String dirRelPath = dataset.relativize(fileDirPath).toString().replace("\\", "/");
        String fileName = file.getName();
        // nginx 直服的相对路径（保留原始文件名）
        String relativePath = dirRelPath + "/" + fileName;

        try (FileInputStream stream = new FileInputStream(file)) {
            String md5 = FileUploadUtils.getMd5(stream);
            String suffix = FileUtil.extName(fileName);
            // nginx URL：http://127.0.0.1:9000/Dense-Haze/clean/01_GT.png
            String url = datasetBaseUrl + "/" + relativePath;

            // 构建 FileBO（objectName 用相对路径，便于定位）
            ItemFileBO itemBO = new ItemFileBO();
            itemBO.setFile(file);
            itemBO.setName(fileName);
            itemBO.setObjectName(relativePath);
            itemBO.setExtension(suffix);
            itemBO.setMd5(md5);
            itemBO.setPath(relativePath);
            itemBO.setSize(file.length());
            itemBO.setUrl(url);
            itemBO.setType(type.getValue());

            // 1. 创建源文件 DB 记录（不上传 MinIO，nginx 直服）
            SysFile sysFile = sysFileService.saveFileRecord(itemBO);

            // 2. 创建 item_file 关联记录（缩略图后续批量生成）
            sysItemFileService.saveItemFileRecord(itemId, itemBO, sysFile, null);

        } catch (IOException e) {
            throw new BusinessException("初始化文件失败: " + filePath, e);
        }
    }

    private ArrayList<PairedImage> getPairedImages(long id) {
        // 获取数据集信息并验证类型
        SysDataset sysDataset = sysDatasetService.getById(id);
        String datasetType = sysDataset.getType();
        if (!DATASET_TYPE_DEHAZE.equals(datasetType)) {
            throw new BusinessException("暂不支持非去雾数据集初始化");
        }

        // 获取数据集路径和文件夹标识
        String filePath = sysDataset.getPath();
        Path datasetBasePath = Path.of(datasetPath, filePath);

        // 获取雾霾图像和清晰图像文件夹路径（大小写不敏感匹配）
        String hazeFlag = getValidPath(HAZE_FOLDER_FLAGS, datasetBasePath);
        String cleanFlag = getValidPath(CLEAN_FOLDER_FLAGS, datasetBasePath);

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

    private void initWpxFile() {
        List<SysWpxFile> sysWpxFiles = sysWpxFileService.list();
        for (SysWpxFile sysWpxFile : sysWpxFiles) {
            // 根据原始图片md5查询原始文件id
            String originMd5 = sysWpxFile.getOriginMd5();
            SysFile originSysFile = sysFileService.getOne(new LambdaQueryWrapper<SysFile>().eq(SysFile::getMd5, originMd5));
            if (originSysFile == null) {
                log.warn("WPX文件初始化跳过: 未找到原始文件 md5={}", originMd5);
                continue;
            }
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
        File[] subDirs = basePath.toFile().listFiles(File::isDirectory);
        if (subDirs == null) {
            return null;
        }
        for (File subDir : subDirs) {
            String dirName = subDir.getName();
            if (flags.stream().anyMatch(flag -> flag.equalsIgnoreCase(dirName))) {
                return dirName;
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

    private static ArrayList<PairedImage> getPairImages(List<String> hazeImages, List<String> cleanImages) {
        if (cleanImages.isEmpty() || hazeImages.isEmpty()) {
            throw new BusinessException("成对图片数量不符：清晰图或雾图列表为空");
        }
        if (hazeImages.size() % cleanImages.size() != 0) {
            throw new BusinessException("成对图片数量不符");
        }
        int hazeCount = hazeImages.size() / cleanImages.size();

        // 构建成对的图像列表
        ArrayList<PairedImage> pairedImageList = new ArrayList<>();
        for (int i = 0; i < cleanImages.size(); i++) {
            List<String> haze = new ArrayList<>();
            String clean = cleanImages.get(i);
            for (int j = 0; j < hazeCount; j++) {
                haze.add(hazeImages.get(i * hazeCount + j));
            }

            PairedImage pairedImage = new PairedImage();
            pairedImage.setCleanPath(clean);
            pairedImage.setHazePath(haze);
            pairedImageList.add(pairedImage);
        }
        return pairedImageList;
    }
}
