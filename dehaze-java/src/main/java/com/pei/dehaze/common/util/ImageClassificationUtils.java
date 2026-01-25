package com.pei.dehaze.common.util;

import cn.hutool.core.text.CharSequenceUtil;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.model.vo.ImageUrlVO;
import lombok.experimental.UtilityClass;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * 图片分类工具类
 * 提供统一的图片类型判断、排序和场景类型提取逻辑
 *
 * @author earthy-zinc
 * @since 2025-01-10
 */
@UtilityClass
public class ImageClassificationUtils {

    /**
     * 图片分类结果
     */
    public static class ClassificationResult {
        private ImageUrlVO clearImage;
        private List<ImageUrlVO> hazyImages;
        private String sceneType;

        public ClassificationResult() {
            this.hazyImages = new ArrayList<>();
        }

        public ImageUrlVO getClearImage() {
            return clearImage;
        }

        public void setClearImage(ImageUrlVO clearImage) {
            this.clearImage = clearImage;
        }

        public List<ImageUrlVO> getHazyImages() {
            return hazyImages;
        }

        public void setHazyImages(List<ImageUrlVO> hazyImages) {
            this.hazyImages = hazyImages;
        }

        public String getSceneType() {
            return sceneType;
        }

        public void setSceneType(String sceneType) {
            this.sceneType = sceneType;
        }
    }

    /**
     * 判断类型字符串是否表示清晰图
     * 支持：clear、clean、清晰、无雾等关键词
     *
     * @param type 类型字符串
     * @return 是否为清晰图
     */
    public static boolean isClearImage(String type) {
        if (type == null) {
            return false;
        }
        String lowerType = type.toLowerCase();
        return lowerType.contains("clear") || lowerType.contains("clean") ||
                type.contains("清晰") || type.contains("无雾");
    }

    /**
     * 判断类型字符串是否表示有雾图
     * 支持：haze、hazy、有雾等关键词
     *
     * @param type 类型字符串
     * @return 是否为有雾图
     */
    public static boolean isHazyImage(String type) {
        if (type == null) {
            return false;
        }
        String lowerType = type.toLowerCase();
        return lowerType.contains("haze") || lowerType.contains("hazy") ||
                type.contains("有雾");
    }

    /**
     * 对ImageUrlVO列表进行分类
     * 包含：清晰图/有雾图分离、按hazeLevel排序、sceneType提取
     *
     * @param images 图片列表
     * @return 分类结果
     */
    public static ClassificationResult classifyImages(List<ImageUrlVO> images) {
        ClassificationResult result = new ClassificationResult();

        if (images == null || images.isEmpty()) {
            return result;
        }

        ImageUrlVO clearImage = null;
        List<ImageUrlVO> hazyImages = new ArrayList<>();

        for (ImageUrlVO image : images) {
            String type = image.getType();
            if (isClearImage(type)) {
                clearImage = image;
            } else if (isHazyImage(type)) {
                hazyImages.add(image);
            }
        }

        // 按雾霾程度排序有雾图
        sortByHazeLevel(hazyImages);

        result.setClearImage(clearImage);
        result.setHazyImages(hazyImages);

        // 提取场景类型：优先从清晰图获取，否则从第一张有雾图获取
        result.setSceneType(extractSceneType(clearImage, hazyImages));

        return result;
    }

    /**
     * 对SysItemFile列表进行分类（通过转换器）
     *
     * @param itemFiles 图片文件列表
     * @param converter 转换器函数
     * @return 分类结果
     */
    public static ClassificationResult classifyItemFiles(
            List<SysItemFile> itemFiles,
            java.util.function.Function<SysItemFile, ImageUrlVO> converter) {

        if (itemFiles == null || itemFiles.isEmpty()) {
            return new ClassificationResult();
        }

        List<ImageUrlVO> images = new ArrayList<>();
        for (SysItemFile itemFile : itemFiles) {
            ImageUrlVO imageVO = converter.apply(itemFile);
            if (imageVO != null) {
                images.add(imageVO);
            }
        }

        return classifyImages(images);
    }

    /**
     * 按雾霾程度排序
     * 排序规则：按严重程度排序（light < medium < heavy），null值排最后
     *
     * @param hazyImages 有雾图列表
     */
    public static void sortByHazeLevel(List<ImageUrlVO> hazyImages) {
        if (hazyImages == null || hazyImages.size() <= 1) {
            return;
        }
        hazyImages.sort(Comparator.comparing(
                ImageUrlVO::getHazeLevel,
                Comparator.nullsLast(Comparator.comparingInt(ImageClassificationUtils::getHazeLevelOrder))
        ));
    }

    /**
     * 获取雾霾程度排序值
     * light=1, medium=2, heavy=3, 其他=99
     *
     * @param level 雾霾程度
     * @return 排序值
     */
    private static int getHazeLevelOrder(String level) {
        if (level == null) {
            return 99;
        }
        return switch (level.toLowerCase()) {
            case "light" -> 1;
            case "medium" -> 2;
            case "heavy" -> 3;
            default -> 99;
        };
    }

    /**
     * 提取场景类型
     * 优先从清晰图获取，否则从第一张有雾图获取
     *
     * @param clearImage 清晰图
     * @param hazyImages 有雾图列表
     * @return 场景类型
     */
    public static String extractSceneType(ImageUrlVO clearImage, List<ImageUrlVO> hazyImages) {
        // 优先从清晰图获取
        if (clearImage != null && CharSequenceUtil.isNotBlank(clearImage.getSceneType())) {
            return clearImage.getSceneType();
        }
        // 其次从第一张有雾图获取
        if (hazyImages != null && !hazyImages.isEmpty()) {
            ImageUrlVO firstHazy = hazyImages.get(0);
            if (firstHazy != null && CharSequenceUtil.isNotBlank(firstHazy.getSceneType())) {
                return firstHazy.getSceneType();
            }
        }
        return null;
    }
}
