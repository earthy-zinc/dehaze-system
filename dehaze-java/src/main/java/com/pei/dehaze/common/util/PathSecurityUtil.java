package com.pei.dehaze.common.util;

import com.pei.dehaze.common.exception.BusinessException;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * 路径安全验证工具类
 * 提供文件路径安全性验证功能，防止路径遍历攻击
 *
 * @author earthy-zinc
 * @since 2026-01-10
 */
public class PathSecurityUtil {

    /**
     * 默认文件名正则表达式：允许字母、数字、点、下划线、连字符
     */
    private static final String DEFAULT_FILENAME_REGEX = "[a-zA-Z0-9.\\-_]+";

    /**
     * 验证文件路径是否安全，防止路径遍历攻击
     *
     * @param fullPath 完整路径（包含基础路径和相对路径）
     * @param basePath 基础路径（允许的根目录）
     * @return 规范化后的绝对路径
     * @throws IllegalArgumentException 当路径不安全时抛出
     */
    public static Path validatePath(Path fullPath, Path basePath) {
        if (fullPath == null) {
            throw new IllegalArgumentException("文件路径不能为null");
        }

        if (basePath == null) {
            throw new IllegalArgumentException("基础路径不能为null");
        }

        // 规范化路径（解析 . 和 ..）
        Path normalizedPath = fullPath.normalize();
        Path normalizedBasePath = basePath.normalize();

        // 验证规范化后的路径是否在基础路径范围内
        if (!normalizedPath.startsWith(normalizedBasePath)) {
            throw new IllegalArgumentException(
                    "无效的文件路径，检测到路径遍历攻击。路径: " + fullPath
            );
        }

        return normalizedPath;
    }

    /**
     * 验证文件路径是否安全，防止路径遍历攻击
     *
     * @param basePathStr  基础路径字符串
     * @param relativePath 相对路径字符串
     * @return 规范化后的绝对路径
     * @throws IllegalArgumentException 当路径不安全时抛出
     */
    public static Path validatePath(String basePathStr, String relativePath) {
        if (basePathStr == null || basePathStr.isBlank()) {
            throw new IllegalArgumentException("基础路径不能为空");
        }

        if (relativePath == null || relativePath.isBlank()) {
            throw new IllegalArgumentException("相对路径不能为空");
        }

        Path basePath = Paths.get(basePathStr);
        Path fullPath = Paths.get(basePathStr, relativePath);

        return validatePath(fullPath, basePath);
    }

    /**
     * 验证文件名是否安全
     * 默认只允许字母、数字、点、下划线、连字符
     *
     * @param fileName 文件名
     * @throws BusinessException 当文件名不安全时抛出
     */
    public static void validateFileName(String fileName) {
        validateFileName(fileName, DEFAULT_FILENAME_REGEX);
    }

    /**
     * 验证文件名是否安全
     *
     * @param fileName 文件名
     * @param regex    文件名正则表达式
     * @throws BusinessException 当文件名不安全时抛出
     */
    public static void validateFileName(String fileName, String regex) {
        if (fileName == null || fileName.isBlank()) {
            throw new BusinessException("文件名不能为空");
        }

        if (!fileName.matches(regex)) {
            throw new BusinessException("不支持的文件名: " + fileName);
        }
    }

    /**
     * 验证文件名是否不包含路径遍历字符
     *
     * @param fileName 文件名
     * @return 是否安全
     */
    public static boolean isFileNameSafe(String fileName) {
        if (fileName == null || fileName.isBlank()) {
            return false;
        }

        // 检查是否包含路径遍历字符
        return !fileName.contains("..") &&
                !fileName.contains("/") &&
                !fileName.contains("\\") &&
                fileName.matches(DEFAULT_FILENAME_REGEX);
    }

    /**
     * 创建安全的文件路径
     * 组合基础路径和相对路径，并验证安全性
     *
     * @param basePath     基础路径
     * @param relativePath 相对路径
     * @return 安全的完整路径
     * @throws IllegalArgumentException 当路径不安全时抛出
     */
    public static Path createSafePath(Path basePath, String relativePath) {
        return validatePath(basePath.resolve(relativePath), basePath);
    }

    /**
     * 获取安全的文件名
     * 移除不安全的字符，只保留字母、数字、点、下划线、连字符
     *
     * @param fileName 原始文件名
     * @return 安全的文件名
     */
    public static String sanitizeFileName(String fileName) {
        if (fileName == null || fileName.isBlank()) {
            return "unnamed";
        }

        // 移除路径分隔符和遍历字符
        String sanitized = fileName.replaceAll("[/\\\\\\.]+", "_");

        // 只保留字母、数字、点、下划线、连字符
        sanitized = sanitized.replaceAll("[^a-zA-Z0-9.\\-_]", "_");

        // 确保不为空
        if (sanitized.isBlank()) {
            return "unnamed";
        }

        return sanitized;
    }
}
