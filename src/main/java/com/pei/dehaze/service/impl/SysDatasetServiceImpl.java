package com.pei.dehaze.service.impl;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.io.FileUtil;
import cn.hutool.core.io.file.PathUtil;
import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.base.BasePageQuery;
import com.pei.dehaze.common.constant.SystemConstants;
import com.pei.dehaze.common.enums.ImageTypeEnum;
import com.pei.dehaze.common.enums.StatusEnum;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.model.Option;
import com.pei.dehaze.common.util.FileUploadUtils;
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
import java.util.ArrayList;
import java.util.Arrays;
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
    public Page<ImageItemVO> getImageItem(Long id, BasePageQuery pageQuery, String hostUrl) {
        Pageable pageable = PageRequest.of(pageQuery.getPageNum() - 1, pageQuery.getPageSize());
        Page<ImageItemVO> pageResult;

        // 获取当前数据集的所有子数据集，从而判断当前数据集是否是叶子数据集
        List<SysDataset> childrenDatasets = this.list(new LambdaQueryWrapper<SysDataset>()
                .eq(SysDataset::getParentId, id));

        if (CollUtil.isEmpty(childrenDatasets)) {
            // 当前为叶子数据集
            pageResult = imageItemRepository.findByDatasetId(id, pageable);
            if (pageResult.getTotalElements() == 0) {
                List<ImageItemVO> imageItemVOS = getImageItemsFromMySQL(id, hostUrl);
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
                    imageItemRepository.saveAll(getImageItemsFromMySQL(childDatasetId, hostUrl));
                }
            });
            return imageItemRepository.findByDatasetIdIn(childrenDatasetIds, pageable);
        }
    }

    @Override
    public boolean addDataset(DatasetForm dataset) {
        SysDataset sysDataset = datasetConverter.form2Entity(dataset);
        sysDataset.setStatus(StatusEnum.ENABLE.getValue());
        Path datasetPath = Path.of(datasetOriginPath, sysDataset.getPath());
        sysDataset.setSize(FileUploadUtils.dirSize(datasetPath));
        sysDataset.setTotal(FileUploadUtils.dirFileCount(datasetPath));
        return this.save(sysDataset);
    }

    @Override
    public boolean updateDataset(DatasetForm dataset) {
        SysDataset sysDataset = datasetConverter.form2Entity(dataset);
        SysDataset old = this.getById(dataset.getId());

        // 获取其绝对路径
        Path datasetPath;
        if (CharSequenceUtil.isNotBlank(sysDataset.getPath())) {
            datasetPath = Path.of(datasetOriginPath, sysDataset.getPath());
        } else {
            datasetPath = Path.of(datasetOriginPath, old.getPath());
        }
        // 更新文件夹大小及文件数量
        if (!sysDataset.getPath().equals(old.getPath())) {
            sysDataset.setSize(FileUploadUtils.dirSize(datasetPath));
            sysDataset.setTotal(FileUploadUtils.dirFileCount(datasetPath));
        }

        sysDataset.setStatus(StatusEnum.ENABLE.getValue());
        return this.updateById(sysDataset);
    }

    @NotNull
    private List<ImageItemVO> getImageItemsFromMySQL(Long id, String hostUrl) {
        List<ImageItemVO> imageItemVOS;
        SysDataset dataset = this.getById(id);
        String datasetType = dataset.getType();
        if (CharSequenceUtil.isNotBlank(datasetType) && datasetType.equals("图像去雾")) {
            imageItemVOS = getImageList(dataset, hostUrl);
        } else {
            imageItemVOS = Collections.emptyList();
        }
        return imageItemVOS;
    }

    private List<ImageItemVO> getImageList(SysDataset dataset, String hostUrl) {
        // 判断当前目录是否存在，然后列出当前目录下所有文件名
        String filePath = dataset.getPath();
        Path datasetBasePath = Path.of(datasetOriginPath, filePath);
        List<String> hazeFlags = Arrays.asList("haze", "hazy");
        List<String> cleanFlags = Arrays.asList("clean", "clear", "gt", "GT");

        String hazeFlag = getValidPath(hazeFlags, datasetBasePath);
        String cleanFlag = getValidPath(cleanFlags, datasetBasePath);

        if (hazeFlag == null || cleanFlag == null) {
            throw new BusinessException("数据集目录" + filePath + "下未找到清晰图像或雾霾图像文件夹");
        }

        Path hazePath = datasetBasePath.resolve(hazeFlag);
        Path cleanPath = datasetBasePath.resolve(cleanFlag);

        if (PathUtil.isDirectory(hazePath) && PathUtil.isDirectory(cleanPath)) {
            List<String> hazeImages = FileUtil.listFileNames(hazePath.toAbsolutePath().toString());
            List<String> cleanImages = FileUtil.listFileNames(cleanPath.toAbsolutePath().toString());
            // 排序和去重
            hazeImages = hazeImages.stream().sorted().distinct().toList();
            cleanImages = cleanImages.stream().sorted().distinct().toList();
            // 相除为大于1的整数
            if (hazeImages.size() % cleanImages.size() == 0) {
                int cleanRepeat = hazeImages.size() / cleanImages.size();
                cleanImages = cleanImages
                        .stream()
                        .flatMap(image -> Collections.nCopies(cleanRepeat, image).stream())
                        .toList();
            }
            return getImageItemVOS(dataset, cleanImages, hazeImages, cleanFlag, hazeFlag, hostUrl);
        }
        throw new BusinessException("数据集目录" + filePath + "下未找到清晰图像或雾霾图像文件夹");
    }

    @NotNull
    private List<ImageItemVO> getImageItemVOS(SysDataset dataset, List<String> cleanImages, List<String> hazeImages,
                                              String cleanFlag, String hazeFlag, String hostUrl) {
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

            List<ImageUrlVO> imageUrls = getImageUrlVOS(dataset, cleanImage, matchedHazeImages, cleanFlag, hazeFlag, hostUrl);

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
                                            String cleanFlag, String hazeFlag, String hostUrl) {
        hostUrl = hostUrl + "/api/v1/files";

        Path datasetPath = Path.of(datasetOriginPath, dataset.getPath());
        if (!PathUtil.isDirectory(datasetPath)) {
            throw new BusinessException("数据集文件夹" + datasetPath + "不存在！");
        }
        String relativeDatasetPath = dataset.getPath().replace("\\", "/");

        List<ImageUrlVO> imageUrls = new ArrayList<>();

        long id = 1L;
        ImageUrlVO cleanImageUrl = new ImageUrlVO();
        cleanImageUrl.setId(id);
        cleanImageUrl.setUrl(hostUrl + "/dataset/thumbnail/" + relativeDatasetPath + "/" + cleanFlag + "/" + cleanImage);
        cleanImageUrl.setOriginUrl(hostUrl + "/dataset/origin/" + relativeDatasetPath + "/" + cleanFlag + "/" + cleanImage);
        cleanImageUrl.setType(ImageTypeEnum.CLEAN.getValue());
        imageUrls.add(cleanImageUrl);
        id++;

        String cleanImageName = cleanImage.substring(0, cleanImage.lastIndexOf("."));
        for (String hazeImage : matchedHazeImages) {
            ImageUrlVO hazeImageUrl = new ImageUrlVO();
            hazeImageUrl.setId(id);
            hazeImageUrl.setUrl(hostUrl + "/dataset/thumbnail/" + relativeDatasetPath + "/" + hazeFlag + "/" + hazeImage);
            hazeImageUrl.setOriginUrl(hostUrl + "/dataset/origin/" + relativeDatasetPath + "/" + hazeFlag + "/" + hazeImage);
            hazeImageUrl.setType(ImageTypeEnum.HAZE.getValue());
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
        // 获取其子数据集id
        List<SysDataset> childDatasets = this.list(new LambdaQueryWrapper<SysDataset>()
                .in(SysDataset::getParentId, ids));
        List<Long> childrenIds = childDatasets.stream().map(SysDataset::getId).toList();
        return this.removeBatchByIds(CollUtil.addAll(ids, childrenIds));
    }

    @Override
    public List<Option<Long>> getOptions() {
        List<SysDataset> datasets = this.list(new LambdaQueryWrapper<>());
        return buildDatasetOptions(SystemConstants.ROOT_NODE_ID, datasets);
    }

    private List<Option<Long>> buildDatasetOptions(Long rootNodeId, List<SysDataset> datasets) {
        List<Option<Long>> options = new ArrayList<>();
        for (SysDataset dataset : datasets) {
            if (dataset.getParentId().equals(rootNodeId)) {
                Option<Long> option = new Option<>(dataset.getId(), dataset.getName());
                List<Option<Long>> subDatasets = buildDatasetOptions(dataset.getId(), datasets);
                if (CollUtil.isNotEmpty(subDatasets)) {
                    option.setChildren(subDatasets);
                }
                options.add(option);
            }
        }
        return options;
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

    private static String getValidPath(List<String> flags, Path basePath) {
        for (String flag : flags) {
            Path path = basePath.resolve(flag);
            if (PathUtil.isDirectory(path)) {
                return flag;
            }
        }
        return null;
    }
}
