package com.pei.dehaze.service.impl;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.io.FileUtil;
import cn.hutool.core.io.file.PathUtil;
import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.base.BasePageQuery;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.util.TreeDataUtils;
import com.pei.dehaze.converter.DatasetConverter;
import com.pei.dehaze.mapper.SysDatasetMapper;
import com.pei.dehaze.model.entity.SysDataset;
import com.pei.dehaze.model.form.DatasetForm;
import com.pei.dehaze.model.query.DatasetQuery;
import com.pei.dehaze.model.vo.DatasetVO;
import com.pei.dehaze.model.vo.ImageItemVO;
import com.pei.dehaze.model.vo.ImageUrlVO;
import com.pei.dehaze.repository.ImageItemRepository;
import com.pei.dehaze.service.SysDatasetService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.jetbrains.annotations.NotNull;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * @author earthy-zinc
 * @since 2024-06-08 18:37:17
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class SysDatasetServiceImpl extends ServiceImpl<SysDatasetMapper, SysDataset> implements SysDatasetService {
    private final DatasetConverter datasetConverter;

    private final ImageItemRepository imageItemRepository;

    @Value("${file.local.datasetOriginPath}")
    private String datasetOriginPath;

    @Value("${file.local.baseUrl}")
    private String baseUrl;

    @Override
    public List<DatasetVO> getList(DatasetQuery queryParams) {
        List<SysDataset> datasets = this.list(new LambdaQueryWrapper<SysDataset>()
                .like(CharSequenceUtil.isNotBlank(queryParams.getKeywords()),
                        SysDataset::getName, queryParams.getKeywords()));

        List<Long> rootIds = TreeDataUtils.findRootIds(datasets, SysDataset::getId, SysDataset::getParentId);

        return rootIds.stream()
                .flatMap(rootId -> buildDatasetTree(rootId, datasets).stream())
                .toList();
    }

    /**
     * 获取数据集内图片
     *
     * @param id 数据集ID
     */
    @Override
    public Page<ImageItemVO> getImageItem(Long id, BasePageQuery pageQuery) {
        Pageable pageable = PageRequest.of(pageQuery.getPageNum(), pageQuery.getPageSize());
        Page<ImageItemVO> pageResult;

        // 获取当前数据集的所有子数据集，从而判断当前数据集是否是叶子数据集
        List<SysDataset> childrenDatasets = this.list(new LambdaQueryWrapper<SysDataset>()
                .eq(SysDataset::getParentId, id));

        if (CollUtil.isEmpty(childrenDatasets)) {
            // 当前为叶子数据集
            pageResult = imageItemRepository.findByDatasetId(id, pageable);
            if (pageResult.getTotalElements() == 0) {
                List<ImageItemVO> imageItemVOS = getImageItemsFromMySQL(id);
                imageItemRepository.saveAll(imageItemVOS);
                return imageItemRepository.findByDatasetId(id, pageable);
            }
            return pageResult;
        } else {
            // 当前为非叶子数据集，则获取其子数据集id列表，从其所有子数据集中查询
            List<Long> childrenDatasetIds = childrenDatasets.stream()
                    .map(SysDataset::getId)
                    .toList();
            childrenDatasetIds.forEach(childDatasetId -> {
                boolean isExist = imageItemRepository.existsByDatasetId(childDatasetId);
                if (!isExist) {
                    imageItemRepository.saveAll(getImageItemsFromMySQL(childDatasetId));
                }
            });
            return imageItemRepository.findByDatasetIdIn(childrenDatasetIds, pageable);
        }
    }

    @Override
    public boolean addDataset(DatasetForm dataset) {
        return false;
    }

    @Override
    public boolean updateDataset(DatasetForm dataset) {
        return false;
    }

    @NotNull
    private List<ImageItemVO> getImageItemsFromMySQL(Long id) {
        List<ImageItemVO> imageItemVOS;
        SysDataset dataset = this.getById(id);
        String datasetType = dataset.getType();
        if (CharSequenceUtil.isNotBlank(datasetType) && datasetType.equals("图像去雾")) {
            imageItemVOS = getImageList(dataset);
        } else {
            imageItemVOS = Collections.emptyList();
        }
        return imageItemVOS;
    }

    private List<ImageItemVO> getImageList(SysDataset dataset) {
        // 判断当前目录是否存在，然后列出当前目录下所有文件名
        String filePath = dataset.getPath();
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
            hazeImages = hazeImages.stream().sorted().distinct().toList();
            cleanImages = cleanImages.stream().sorted().distinct().toList();

            return getImageItemVOS(dataset, cleanImages, hazeImages, cleanFlag, hazeFlag);
        }
        throw new BusinessException("数据集目录" + filePath + "下未找到清晰图像或雾霾图像文件夹");
    }

    @NotNull
    private List<ImageItemVO> getImageItemVOS(SysDataset dataset, List<String> cleanImages, List<String> hazeImages,
                                              String cleanFlag, String hazeFlag) {
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

            List<ImageUrlVO> imageUrls = getImageUrlVOS(dataset, cleanImage, matchedHazeImages, cleanFlag, hazeFlag);

            ImageItemVO imageItemVO = new ImageItemVO();
            imageItemVO.setId((long) i);
            imageItemVO.setDatasetId(dataset.getId());
            imageItemVO.setImgUrl(imageUrls);

            imageItemVOs.add(imageItemVO);
        }
        return imageItemVOs;
    }

    @NotNull
    private List<ImageUrlVO> getImageUrlVOS(SysDataset dataset, String cleanImage, List<String> matchedHazeImages,
                                            String cleanFlag, String hazeFlag) {
        Path baseDirPath = Paths.get(datasetOriginPath);
        Path datasetPath = Paths.get(dataset.getPath());
        if (!PathUtil.isSub(baseDirPath, datasetPath)) {
            throw new BusinessException("数据集目录" + datasetPath + "不在" + baseDirPath + "目录下");
        }
        String relativeDatasetPath = baseDirPath.relativize(datasetPath).toString().replace("\\", "/");

        List<ImageUrlVO> imageUrls = new ArrayList<>();

        long id = 1L;
        ImageUrlVO cleanImageUrl = new ImageUrlVO();
        cleanImageUrl.setId(id);
        cleanImageUrl.setUrl(baseUrl + "/dataset/thumbnail/" + relativeDatasetPath + "/" + cleanFlag + "/" + cleanImage);
        cleanImageUrl.setOriginUrl(baseUrl + "/dataset/origin/" + relativeDatasetPath + "/" + cleanFlag + "/" + cleanImage);
        cleanImageUrl.setType("clean");
        imageUrls.add(cleanImageUrl);
        id++;

        String cleanImageName = cleanImage.substring(0, cleanImage.lastIndexOf("."));
        for (String hazeImage : matchedHazeImages) {
            ImageUrlVO hazeImageUrl = new ImageUrlVO();
            hazeImageUrl.setId(id);
            hazeImageUrl.setUrl(baseUrl + "/dataset/thumbnail/" + relativeDatasetPath + "/" + hazeFlag + "/" + hazeImage);
            hazeImageUrl.setOriginUrl(baseUrl + "/dataset/origin/" + relativeDatasetPath + "/" + hazeFlag + "/" + hazeImage);
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
