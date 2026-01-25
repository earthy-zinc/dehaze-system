package com.pei.dehaze.common.util;

import cn.hutool.core.date.DateUtil;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;

/**
 * 统一文件路径构建工具
 * <p>
 * 收敛所有文件路径构建逻辑，避免路径规则散落在 Controller、Service、工具类中
 *
 * @author earthy-zinc
 * @since 2025-01-19
 */
@Component
public class FilePathBuilder {

    private static final String UPLOAD_PATH_FORMAT = "upload/%s";
    private static final String THUMBNAIL_PATH_FORMAT = "thumbnail/%s/%s.%s";
    private static final String EXPORT_PATH_FORMAT = "exports/%s.zip";
    private static final String DATE_FORMAT = "yyyyMMdd";

    @Value("${file.baseUrl}")
    private String baseUrl;

    /**
     * 构建当日上传路径
     *
     * @return 上传路径，格式：upload/yyyyMMdd
     */
    public String buildUploadPath() {
        return buildUploadPath(LocalDateTime.now());
    }

    /**
     * 构建指定日期的上传路径
     *
     * @param dateTime 日期时间
     * @return 上传路径，格式：upload/yyyyMMdd
     */
    public String buildUploadPath(LocalDateTime dateTime) {
        String date = DateUtil.format(dateTime, DATE_FORMAT);
        return String.format(UPLOAD_PATH_FORMAT, date);
    }

    /**
     * 构建对象名（存储路径）
     *
     * @param date      日期字符串（yyyyMMdd）
     * @param md5       文件 MD5
     * @param extension 文件扩展名
     * @return objectName，格式：upload/yyyyMMdd/md5.ext
     */
    public String buildObjectName(String date, String md5, String extension) {
        return String.format("upload/%s/%s.%s", date, md5, extension);
    }

    /**
     * 构建对象名（使用当前日期）
     *
     * @param md5       文件 MD5
     * @param extension 文件扩展名
     * @return objectName
     */
    public String buildObjectName(String md5, String extension) {
        String date = DateUtil.format(LocalDateTime.now(), DATE_FORMAT);
        return buildObjectName(date, md5, extension);
    }

    /**
     * 构建缩略图路径
     *
     * @param originPath 原图路径（不含文件名）
     * @param md5        缩略图 MD5
     * @param extension  文件扩展名
     * @return 缩略图路径，格式：thumbnail/originPath/md5.ext
     */
    public String buildThumbnailPath(String originPath, String md5, String extension) {
        return String.format(THUMBNAIL_PATH_FORMAT, originPath, md5, extension);
    }

    /**
     * 构建缩略图对象名
     *
     * @param originObjectName 原图对象名
     * @param thumbnailMd5     缩略图 MD5
     * @param extension        文件扩展名
     * @return 缩略图对象名
     */
    public String buildThumbnailObjectName(String originObjectName, String thumbnailMd5, String extension) {
        // 从原图对象名中提取路径部分（去掉文件名）
        int lastSlashIndex = originObjectName.lastIndexOf('/');
        String originPath = lastSlashIndex > 0 ? originObjectName.substring(0, lastSlashIndex) : originObjectName;
        return String.format("thumbnail/%s/%s.%s", originPath, thumbnailMd5, extension);
    }

    /**
     * 构建导出文件路径
     *
     * @param taskId 任务ID
     * @return 导出路径，格式：exports/taskId.zip
     */
    public String buildExportPath(String taskId) {
        return String.format(EXPORT_PATH_FORMAT, taskId);
    }

    /**
     * 构建文件访问 URL
     *
     * @param objectName 对象名
     * @return 完整 URL
     */
    public String buildUrl(String objectName) {
        return baseUrl + "/" + objectName;
    }

    /**
     * 获取基础 URL
     *
     * @return baseUrl
     */
    public String getBaseUrl() {
        return baseUrl;
    }
}
