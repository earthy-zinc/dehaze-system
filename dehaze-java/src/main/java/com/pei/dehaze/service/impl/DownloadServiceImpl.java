package com.pei.dehaze.service.impl;

import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.model.entity.SysFile;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.model.vo.DownloadTaskVO;
import com.pei.dehaze.service.DownloadService;
import com.pei.dehaze.service.SysDatasetService;
import com.pei.dehaze.service.SysFileService;
import com.pei.dehaze.service.SysItemFileService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/**
 * 下载服务实现类
 *
 * @author earthy-zinc
 * @since 2025-12-07
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class DownloadServiceImpl implements DownloadService {

    private final RedisTemplate<String, Object> redisTemplate;
    private final SysItemFileService sysItemFileService;
    private final SysDatasetService sysDatasetService;
    private final SysFileService sysFileService;

    // 线程上下文，用于传递任务ID
    private static final ThreadLocal<String> currentTaskId = new ThreadLocal<>();

    private static final String DOWNLOAD_TASK_KEY_PREFIX = "download:task:";
    private static final String DOWNLOAD_URL_PREFIX = "/api/v1/download/";
    private static final int CACHE_EXPIRE_HOURS = 24;

    @Override
    public String createDatasetDownloadTask(Long datasetId, boolean organizeByItem) {
        String taskId = UUID.randomUUID().toString();

        // 初始化任务状态
        DownloadTaskVO task = new DownloadTaskVO();
        task.setTaskId(taskId);
        task.setStatus("processing");
        task.setProgress(0);
        task.setMessage("正在创建下载任务...");
        task.setCreateTime(LocalDateTime.now());

        // 缓存任务状态
        redisTemplate.opsForValue().set(
                DOWNLOAD_TASK_KEY_PREFIX + taskId,
                task,
                CACHE_EXPIRE_HOURS,
                TimeUnit.HOURS
        );

        // 异步处理任务
        currentTaskId.set(taskId);
        processDatasetDownloadTask(taskId, datasetId, organizeByItem);
        currentTaskId.remove();

        return taskId;
    }

    @Override
    public String createBatchImageItemDownloadTask(List<Long> itemFileIds, boolean organizeByItem) {
        // 限制批量下载数量
        if (itemFileIds != null && itemFileIds.size() > 500) {
            throw new BusinessException("单次下载不能超过500张图片");
        }

        String taskId = UUID.randomUUID().toString();

        // 初始化任务状态
        DownloadTaskVO task = new DownloadTaskVO();
        task.setTaskId(taskId);
        task.setStatus("processing");
        task.setProgress(0);
        task.setMessage("正在创建下载任务...");
        task.setCreateTime(LocalDateTime.now());

        // 缓存任务状态
        redisTemplate.opsForValue().set(
                DOWNLOAD_TASK_KEY_PREFIX + taskId,
                task,
                CACHE_EXPIRE_HOURS,
                TimeUnit.HOURS
        );

        // 异步处理任务
        currentTaskId.set(taskId);
        processBatchImageDownloadTask(taskId, itemFileIds, organizeByItem);
        currentTaskId.remove();

        return taskId;
    }

    @Override
    @Async
    public void processDatasetDownloadTask(String taskId, Long datasetId, boolean organizeByItem) {
        currentTaskId.set(taskId);
        try {
            updateTaskProgress(taskId, 10, "正在获取数据集图片列表...");

            // 获取数据集下所有图片文件
            List<SysItemFile> itemFiles = sysDatasetService.getDatasetImages(datasetId, true);

            // 检查数据集大小
            if (itemFiles.size() > 1000) {
                throw new BusinessException("数据集过大，图片数量超过1000张，请使用分批下载");
            }

            updateTaskProgress(taskId, 30, "正在打包文件...");

            // 创建ZIP文件
            String zipFilePath = createZipFile(itemFiles, organizeByItem);

            updateTaskProgress(taskId, 100, "下载完成");

            // 更新任务状态为完成
            DownloadTaskVO task = getTaskStatus(taskId);
            task.setStatus("completed");
            task.setProgress(100);
            task.setMessage("下载完成");
            task.setDownloadUrl(DOWNLOAD_URL_PREFIX + Paths.get(zipFilePath).getFileName().toString());
            task.setExpireTime(LocalDateTime.now().plusHours(CACHE_EXPIRE_HOURS));

            redisTemplate.opsForValue().set(
                    DOWNLOAD_TASK_KEY_PREFIX + taskId,
                    task,
                    CACHE_EXPIRE_HOURS,
                    TimeUnit.HOURS
            );

        } catch (Exception e) {
            log.error("处理数据集下载任务失败", e);
            updateTaskFailed(taskId, "打包失败: " + e.getMessage());
        } finally {
            currentTaskId.remove();
        }
    }

    @Override
    @Async
    public void processBatchImageDownloadTask(String taskId, List<Long> itemFileIds, boolean organizeByItem) {
        currentTaskId.set(taskId);
        try {
            updateTaskProgress(taskId, 10, "正在获取图片信息...");

            // 获取指定图片文件
            List<SysItemFile> itemFiles = sysItemFileService.listByIds(itemFileIds);

            updateTaskProgress(taskId, 30, "正在打包文件...");

            // 创建ZIP文件
            String zipFilePath = createZipFile(itemFiles, organizeByItem);

            updateTaskProgress(taskId, 100, "下载完成");

            // 更新任务状态为完成
            DownloadTaskVO task = getTaskStatus(taskId);
            task.setStatus("completed");
            task.setProgress(100);
            task.setMessage("下载完成");
            task.setDownloadUrl(DOWNLOAD_URL_PREFIX + Paths.get(zipFilePath).getFileName().toString());
            task.setExpireTime(LocalDateTime.now().plusHours(CACHE_EXPIRE_HOURS));

            redisTemplate.opsForValue().set(
                    DOWNLOAD_TASK_KEY_PREFIX + taskId,
                    task,
                    CACHE_EXPIRE_HOURS,
                    TimeUnit.HOURS
            );

        } catch (Exception e) {
            log.error("处理批量图片下载任务失败", e);
            updateTaskFailed(taskId, "打包失败: " + e.getMessage());
        } finally {
            currentTaskId.remove();
        }
    }

    @Override
    public DownloadTaskVO getTaskStatus(String taskId) {
        return (DownloadTaskVO) redisTemplate.opsForValue().get(DOWNLOAD_TASK_KEY_PREFIX + taskId);
    }

    /**
     * 创建ZIP文件
     *
     * @param itemFiles     图片文件列表
     * @param organizeByItem 是否按数据项分目录组织
     * @return ZIP文件路径
     * @throws IOException IO异常
     */
    private String createZipFile(List<SysItemFile> itemFiles, boolean organizeByItem) throws IOException {
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        String zipFileName = "download_" + timestamp + ".zip";
        String tempDir = System.getProperty("java.io.tmpdir");
        String zipFilePath = tempDir + File.separator + zipFileName;

        try (FileOutputStream fos = new FileOutputStream(zipFilePath);
             ZipOutputStream zos = new ZipOutputStream(fos)) {

            for (int i = 0; i < itemFiles.size(); i++) {
                SysItemFile itemFile = itemFiles.get(i);

                // 获取文件路径
                SysFile fileEntity = sysFileService.getById(itemFile.getFileId());
                if (fileEntity == null) {
                    log.warn("文件不存在，fileId: {}", itemFile.getFileId());
                    continue;
                }

                // 构建完整的文件路径
                String filePath = fileEntity.getPath();
                if (filePath == null || filePath.isEmpty()) {
                    log.warn("文件路径为空，fileId: {}", itemFile.getFileId());
                    continue;
                }

                // 如果是相对路径，需要构建完整路径
                File file;
                if (filePath.startsWith("/")) {
                    file = new File(filePath);
                } else {
                    // 对于相对路径，假设存储在配置的baseUrl目录下
                    file = new File(filePath);
                }

                if (!file.exists()) {
                    log.warn("文件不存在: {}", filePath);
                    continue;
                }

                // 确定ZIP中的文件路径
                String zipEntryPath;
                if (organizeByItem) {
                    // 按数据项分目录组织
                    String itemDir = "item_" + itemFile.getItemId();
                    zipEntryPath = itemDir + File.separator + fileEntity.getName();
                } else {
                    zipEntryPath = fileEntity.getName();
                }

                // 添加文件到ZIP
                addToZip(file, zipEntryPath, zos);

                // 更新进度
                int progress = 30 + (int) (60.0 * (i + 1) / itemFiles.size());
                updateTaskProgress(getCurrentTaskId(), progress, "正在打包文件... (" + (i + 1) + "/" + itemFiles.size() + ")");
            }
        }

        return zipFilePath;
    }

    /**
     * 添加文件到ZIP
     *
     * @param file        源文件
     * @param entryPath   ZIP中的路径
     * @param zos         ZIP输出流
     * @throws IOException IO异常
     */
    private void addToZip(File file, String entryPath, ZipOutputStream zos) throws IOException {
        try (FileInputStream fis = new FileInputStream(file)) {
            ZipEntry zipEntry = new ZipEntry(entryPath);
            zos.putNextEntry(zipEntry);

            byte[] buffer = new byte[1024];
            int length;
            while ((length = fis.read(buffer)) > 0) {
                zos.write(buffer, 0, length);
            }

            zos.closeEntry();
        }
    }

    /**
     * 更新任务进度
     *
     * @param taskId 任务ID
     * @param progress 进度百分比
     * @param message 消息
     */
    private void updateTaskProgress(String taskId, int progress, String message) {
        DownloadTaskVO task = getTaskStatus(taskId);
        if (task != null) {
            task.setProgress(progress);
            task.setMessage(message);
            redisTemplate.opsForValue().set(
                    DOWNLOAD_TASK_KEY_PREFIX + taskId,
                    task,
                    CACHE_EXPIRE_HOURS,
                    TimeUnit.HOURS
            );
        }
    }

    /**
     * 更新任务失败状态
     *
     * @param taskId 任务ID
     * @param message 失败消息
     */
    private void updateTaskFailed(String taskId, String message) {
        DownloadTaskVO task = getTaskStatus(taskId);
        if (task != null) {
            task.setStatus("failed");
            task.setMessage(message);
            redisTemplate.opsForValue().set(
                    DOWNLOAD_TASK_KEY_PREFIX + taskId,
                    task,
                    CACHE_EXPIRE_HOURS,
                    TimeUnit.HOURS
            );
        }
    }

    /**
     * 获取当前任务ID
     */
    private String getCurrentTaskId() {
        return currentTaskId.get();
    }
}
