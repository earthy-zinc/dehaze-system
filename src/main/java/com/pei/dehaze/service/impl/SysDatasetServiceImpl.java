package com.pei.dehaze.service.impl;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.constant.SystemConstants;
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
import com.pei.dehaze.service.SysDatasetService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.nio.file.Path;
import java.util.ArrayList;
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

    @Value("${file.datasetPath}")
    private String datasetPath;

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

    @Override
    public boolean addDataset(DatasetForm dataset) {
        SysDataset sysDataset = datasetConverter.form2Entity(dataset);
        sysDataset.setStatus(StatusEnum.ENABLE.getValue());
        Path path = Path.of(this.datasetPath, sysDataset.getPath());
        sysDataset.setSize(FileUploadUtils.dirSize(path));
        return this.save(sysDataset);
    }

    @Override
    public boolean updateDataset(DatasetForm dataset) {
        SysDataset sysDataset = datasetConverter.form2Entity(dataset);
        SysDataset old = this.getById(dataset.getId());

        // 获取其绝对路径
        Path path;
        if (CharSequenceUtil.isNotBlank(sysDataset.getPath())) {
            path = Path.of(this.datasetPath, sysDataset.getPath());
        } else {
            path = Path.of(this.datasetPath, old.getPath());
        }
        // 更新文件夹大小及文件数量
        if (!sysDataset.getPath().equals(old.getPath())) {
            sysDataset.setSize(FileUploadUtils.dirSize(path));
        }

        sysDataset.setStatus(StatusEnum.ENABLE.getValue());
        return this.updateById(sysDataset);
    }


    @Override
    @Transactional
    public boolean deleteDatasets(List<Long> ids) {
        // 获取其子数据集id
        List<SysDataset> childDatasets = this.list(new LambdaQueryWrapper<SysDataset>()
                .in(SysDataset::getParentId, ids));
        List<Long> childrenIds = childDatasets.stream().map(SysDataset::getId).toList();
        this.baseMapper.deleteBatchIds(CollUtil.addAll(ids, childrenIds));
        return true;
    }

    @Override
    public List<Option<Long>> getOptions() {
        List<SysDataset> datasets = this.list(new LambdaQueryWrapper<>());
        return buildDatasetOptions(SystemConstants.ROOT_NODE_ID, datasets);
    }

    @Override
    public List<Long> getLeafDatasetIds() {
        List<SysDataset> datasets = this.list();
        return datasets.stream()
                .filter(dataset -> {
                    List<SysDataset> child = this.list(new LambdaQueryWrapper<SysDataset>()
                            .eq(SysDataset::getParentId, dataset.getId()));
                    return child.isEmpty();
                })
                .map(SysDataset::getId)
                .toList();
    }

    @Override
    public SysDataset getRootDataset(Long id) {
        List<SysDataset> datasets = new ArrayList<>();
        // 获取当前节点
        SysDataset cur = this.getById(id);
        // 执行深度优先遍历
        dfs(cur, datasets);

        // 将子节点的 name 累加到父节点的 name 中
        if (!datasets.isEmpty()) {
            SysDataset root = datasets.get(datasets.size() - 1);
            // 将所有子节点的 name 追加到根节点的 name
            StringBuilder fullName = new StringBuilder(root.getName());
            StringBuilder fullDescription = new StringBuilder(root.getDescription());
            for (int i = datasets.size() - 2; i >= 0; i--) {
                SysDataset dataset = datasets.get(i);
                fullName.append("/").append(dataset.getName()); // 这里你可以自定义分隔符
                fullDescription.append("\n").append(dataset.getDescription());
            }
            root.setName(fullName.toString());
            root.setDescription(fullDescription.toString());
            return root;
        }
        return null;
    }

    @Override
    public SysDataset getDatasetById(Long id) {
        List<SysDataset> datasets = new ArrayList<>();
        // 获取当前节点并执行深度优先遍历
        SysDataset cur = this.getById(id);
        dfs(cur, datasets);

        // 将祖先节点的 name 按照顺序追加到当前节点的 name 前面
        if (!datasets.isEmpty()) {
            StringBuilder fullName = new StringBuilder();
            StringBuilder fullDescription = new StringBuilder();
            // 从 root 到当前节点，依次追加每个节点的 name
            for (int i = datasets.size() - 1; i >= 0; i--) { // 从根节点到当前节点
                fullName.append(datasets.get(i).getName()).append("/");
                fullDescription.append(datasets.get(i).getDescription()).append("\n");
            }
            // 移除最后一个不必要的 " -> "
            if (fullName.length() > 4) {
                fullName.setLength(fullName.length() - 1);
            }
            // 设置当前节点的 name
            cur.setName(fullName.toString());
            cur.setDescription(fullDescription.toString());
        }
        return cur;
    }

    private void dfs(SysDataset cur, List<SysDataset> datasets) {
        datasets.add(cur); // 将当前节点加入结果列表
        if (cur.getParentId() == null) {
            throw new BusinessException("数据集结构出现问题");
        }
        if (!cur.getParentId().equals(SystemConstants.ROOT_NODE_ID)) {
            // 获取父节点并递归
            SysDataset parent = this.getById(cur.getParentId());
            dfs(parent, datasets);
        }
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
}
