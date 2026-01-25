package com.pei.dehaze.service;

import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.util.Set;

/**
 * 图片处理服务接口
 * <p>
 * 统一图片校验、缩略图生成、宽高解析等逻辑，
 * 从 SysItemFileServiceImpl 和 FileUploadUtils 抽取
 *
 * @author earthy-zinc
 * @since 2025-01-19
 */
public interface ImageProcessingService {

    /**
     * 校验图片文件（格式、大小等）
     *
     * @param file 图片文件
     * @throws com.pei.dehaze.common.exception.BusinessException 校验失败时抛出
     */
    void validateImageFile(File file);

    /**
     * 校验图片文件（MultipartFile 版本）
     *
     * @param file 上传的图片文件
     * @throws com.pei.dehaze.common.exception.BusinessException 校验失败时抛出
     */
    void validateImageFile(MultipartFile file);

    /**
     * 生成缩略图
     *
     * @param source   源文件
     * @param maxWidth 最大宽度
     * @return 缩略图文件（临时文件）
     */
    File generateThumbnail(File source, int maxWidth);

    /**
     * 生成缩略图（指定宽高）
     *
     * @param source 源文件
     * @param width  目标宽度
     * @param height 目标高度
     * @return 缩略图文件（临时文件）
     */
    File generateThumbnail(File source, int width, int height);

    /**
     * 生成缩略图到指定路径
     *
     * @param srcPath  源文件路径
     * @param destPath 目标路径
     * @param width    目标宽度
     * @param height   目标高度
     */
    void generateThumbnail(String srcPath, String destPath, int width, int height);

    /**
     * 解析图片宽高
     *
     * @param file 图片文件
     * @return int数组，[0]为宽度，[1]为高度，解析失败返回[0,0]
     */
    int[] getImageDimensions(File file);

    /**
     * 解析图片宽高（MultipartFile 版本）
     *
     * @param file 上传的图片文件
     * @return int数组，[0]为宽度，[1]为高度，解析失败返回[0,0]
     */
    int[] getImageDimensions(MultipartFile file);

    /**
     * 判断是否为支持的图片格式
     *
     * @param extension 文件扩展名（不含点）
     * @return 是否支持
     */
    boolean isSupportedImageFormat(String extension);

    /**
     * 判断文件名是否为图片
     *
     * @param fileName 文件名
     * @return 是否为图片
     */
    boolean isImage(String fileName);

    /**
     * 获取支持的图片格式列表
     *
     * @return 支持的格式集合
     */
    Set<String> getSupportedFormats();

    /**
     * 获取最大文件大小限制（字节）
     *
     * @return 最大文件大小
     */
    long getMaxFileSize();
}
