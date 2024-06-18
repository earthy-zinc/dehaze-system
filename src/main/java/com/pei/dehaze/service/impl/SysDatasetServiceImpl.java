package com.pei.dehaze.service.impl;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.io.FileUtil;
import cn.hutool.core.io.file.PathUtil;
import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.converter.DatasetConverter;
import com.pei.dehaze.mapper.SysDatasetMapper;
import com.pei.dehaze.model.entity.SysDataset;
import com.pei.dehaze.model.query.DatasetQuery;
import com.pei.dehaze.model.vo.DatasetVO;
import com.pei.dehaze.model.vo.ImageItemVO;
import com.pei.dehaze.model.vo.ImageUrlVO;
import com.pei.dehaze.service.SysDatasetService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.jetbrains.annotations.NotNull;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * @author earthy-zinc
 * @since 2024-06-08 18:37:17
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class SysDatasetServiceImpl extends ServiceImpl<SysDatasetMapper, SysDataset> implements SysDatasetService {
    private final DatasetConverter datasetConverter;

    @Value("${file.local.baseUrl}")
    private String baseUrl;

    @Override
    public List<DatasetVO> getList(DatasetQuery queryParams) {
        List<SysDataset> datasets = this.list(new LambdaQueryWrapper<SysDataset>()
                .like(CharSequenceUtil.isNotBlank(queryParams.getKeywords()), SysDataset::getName, queryParams.getKeywords()));

        Set<Long> datasetIds = datasets.stream()
                .map(SysDataset::getId)
                .collect(Collectors.toSet());

        Set<Long> parentIds = datasets.stream()
                .map(SysDataset::getParentId)
                .collect(Collectors.toSet());

        List<Long> rootIds = parentIds.stream()
                .filter(id -> !datasetIds.contains(id))
                .toList();

        return rootIds.stream()
                .flatMap(rootId -> buildDatasetTree(rootId, datasets).stream())
                .collect(Collectors.toList());
    }

    /**
     * 获取数据集内图片
     *
     * @param id 数据集ID
     */
    @Override
    public List<ImageItemVO> getImageItem(Long id) {
        SysDataset dataset = this.getById(id);
        String datasetType = dataset.getType();
        if (CharSequenceUtil.isNotBlank(datasetType) && datasetType.equals("图像去雾")) {
            String filePath = dataset.getPath();
            String datasetName = dataset.getName();
            return getImageList(filePath, datasetName);
        }
        return Collections.emptyList();
    }

    private List<ImageItemVO> getImageList(String filePath, String datasetName) {
        // 判断当前目录是否存在，然后列出当前目录下所有文件名
        Path datasetBasePath = Paths.get(filePath);

        String hazeFlag = "haze";
        Path hazePath = datasetBasePath.resolve(hazeFlag);
        if (!PathUtil.isDirectory(hazePath)) {
            hazeFlag = "hazy";
            hazePath = datasetBasePath.resolve(hazeFlag);
        }
        String cleanFlag = "clean";
        Path cleanPath = datasetBasePath.resolve(cleanFlag);
        if (!PathUtil.isDirectory(cleanPath)) {
            cleanFlag = "gt";
            cleanPath = datasetBasePath.resolve(cleanFlag);
        }

        if (PathUtil.isDirectory(hazePath) && PathUtil.isDirectory(cleanPath)) {
            List<String> hazeImages = FileUtil.listFileNames(hazePath.toAbsolutePath().toString());
            List<String> cleanImages = FileUtil.listFileNames(cleanPath.toAbsolutePath().toString());
            // 排序和去重
            hazeImages = hazeImages.stream().sorted().distinct().collect(Collectors.toList());
            cleanImages = cleanImages.stream().sorted().distinct().toList();

            List<ImageItemVO> imageItemVOs = new ArrayList<>();
            for (int i = 0; i < cleanImages.size(); i++) {
                String cleanImage = cleanImages.get(i);
                List<String> matchedHazeImages = new ArrayList<>();

                if (cleanImages.size() == hazeImages.size()) {
                    matchedHazeImages.add(hazeImages.get(i));
                } else {
                    String cleanImageNamePrefix;
                    String cleanImageName = cleanImage.substring(0, cleanImage.lastIndexOf("."));
                    if (cleanImageName.contains("_")) {
                        cleanImageNamePrefix = cleanImage.split("_")[0];
                    } else {
                        cleanImageNamePrefix = cleanImageName;
                    }
                    hazeImages.removeIf(image -> {
                        if (image.startsWith(cleanImageNamePrefix)) {
                            matchedHazeImages.add(image);
                            return true;
                        }
                        // 当遇到第一个不匹配的元素时停止遍历，因为列表是排序的，后续也不会匹配
                        return image.compareTo(cleanImageNamePrefix) > 0;
                    });
                }

                List<ImageUrlVO> imageUrls = getImageUrlVOS(datasetName, cleanImage, matchedHazeImages, cleanFlag, hazeFlag);

                ImageItemVO imageItemVO = new ImageItemVO();
                imageItemVO.setId((long) i);
                imageItemVO.setImgUrl(imageUrls);

                imageItemVOs.add(imageItemVO);
            }
            return imageItemVOs;
        }
        throw new BusinessException("数据集目录" + filePath + "下未找到清晰图像或雾霾图像文件夹");
    }

    @NotNull
    private List<ImageUrlVO> getImageUrlVOS(String datasetName, String cleanImage, List<String> matchedHazeImages, String cleanFlag, String hazeFlag) {
        List<ImageUrlVO> imageUrls = new ArrayList<>();

        long id = 1L;
        ImageUrlVO cleanImageUrl = new ImageUrlVO();
        cleanImageUrl.setId(id);
        cleanImageUrl.setUrl(baseUrl + "/dataset/thumbnail/" + datasetName + "/" + cleanFlag + "/" + cleanImage);
        cleanImageUrl.setOriginUrl(baseUrl + "/dataset/origin/" + datasetName + "/" + cleanFlag + "/" + cleanImage);
        cleanImageUrl.setType("clean");
        imageUrls.add(cleanImageUrl);
        id++;

        String cleanImageName = cleanImage.substring(0, cleanImage.lastIndexOf("."));
        for (String hazeImage : matchedHazeImages) {
            ImageUrlVO hazeImageUrl = new ImageUrlVO();
            hazeImageUrl.setId(id);
            hazeImageUrl.setUrl(baseUrl + "/dataset/thumbnail/" + datasetName + "/" + hazeFlag + "/" + hazeImage);
            hazeImageUrl.setOriginUrl(baseUrl + "/dataset/origin/" + datasetName + "/" + hazeFlag + "/" + hazeImage);
            hazeImageUrl.setType("haze");
            if (hazeImage.startsWith(cleanImageName)) {
                int prefixLength = cleanImageName.length();
                String description = hazeImage.substring(prefixLength);
                hazeImageUrl.setDescription(description);
            }
            imageUrls.add(hazeImageUrl);
            id++;
        }
        return imageUrls;
    }

    @Override
    public boolean deleteDatasets(List<Long> ids) {
        return this.removeByIds(ids);
    }

    private List<DatasetVO> buildDatasetTree(Long rootId, List<SysDataset> datasets) {
        return CollUtil.emptyIfNull(datasets)
                .stream()
                .filter(dataset -> dataset.getParentId().equals(rootId))
                .map(entity -> {
                    DatasetVO datasetVO = datasetConverter.entity2Vo(entity);
                    datasetVO.setChildren(buildDatasetTree(entity.getId(), datasets));
                    return datasetVO;
                }).toList();
    }
}
