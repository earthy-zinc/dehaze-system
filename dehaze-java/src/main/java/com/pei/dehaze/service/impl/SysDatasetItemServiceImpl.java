package com.pei.dehaze.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.mapper.SysDatasetItemMapper;
import com.pei.dehaze.model.entity.SysDatasetItem;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.model.vo.ImageItemVO;
import com.pei.dehaze.model.vo.ImageUrlVO;
import com.pei.dehaze.service.SysDatasetItemService;
import com.pei.dehaze.service.SysDatasetService;
import com.pei.dehaze.service.SysItemFileService;
import jakarta.annotation.Resource;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class SysDatasetItemServiceImpl extends ServiceImpl<SysDatasetItemMapper, SysDatasetItem>
        implements SysDatasetItemService {

    @Resource
    private SysItemFileService sysItemFileService;

    @Resource
    private SysDatasetService sysDatasetService;

    @Override
    public SysDatasetItem createDatasetItem(Long datasetId) {
        SysDatasetItem datasetItem = new SysDatasetItem();
        datasetItem.setDatasetId(datasetId);
        this.save(datasetItem);
        return datasetItem;
    }

    @Override
    public SysDatasetItem createDatasetItem(Long datasetId, String itemName) {
        SysDatasetItem datasetItem = new SysDatasetItem();
        datasetItem.setDatasetId(datasetId);
        datasetItem.setName(itemName);
        this.save(datasetItem);
        return datasetItem;
    }

    @Override
    public void deleteDatasetItem(Long datasetItemId) {
        List<SysItemFile> list = sysItemFileService.list(new LambdaQueryWrapper<SysItemFile>().eq(SysItemFile::getItemId, datasetItemId));
        list.stream().map(SysItemFile::getId).forEach(id -> sysItemFileService.deleteItemFile(id));
        this.removeById(datasetItemId);
    }

    @Override
    public void updateDatasetItem(Long datasetItemId, String itemName) {
        SysDatasetItem datasetItem = this.getById(datasetItemId);
        datasetItem.setName(itemName);
        this.updateById(datasetItem);
    }

    @Override
    public Page<ImageItemVO> getPagedImageItemVOs(Long datasetId, int pageNum, int pageSize) {
        List<Long> leafIds = sysDatasetService.getLeafDatasetId(datasetId);

        Page<SysDatasetItem> page = this.page(
                new Page<>(pageNum, pageSize),
                new LambdaQueryWrapper<SysDatasetItem>()
                        .in(SysDatasetItem::getDatasetId, leafIds));

        List<SysDatasetItem> list = page.getRecords();
        List<ImageItemVO> imageItemVOS = list.stream().map(item -> {
            List<ImageUrlVO> imageUrlVOs = sysItemFileService.getImageUrlVOs(item.getId());
            ImageItemVO imageItemVO = new ImageItemVO();
            imageItemVO.setId(item.getId());
            imageItemVO.setDatasetId(item.getDatasetId());
            imageItemVO.setImgUrl(imageUrlVOs);
            return imageItemVO;
        }).toList();
        Page<ImageItemVO> imageItemVOPage = new Page<>();
        imageItemVOPage.setRecords(imageItemVOS);
        imageItemVOPage.setTotal(page.getTotal());
        imageItemVOPage.setCurrent(page.getCurrent());
        imageItemVOPage.setSize(page.getSize());
        return imageItemVOPage;
    }
}
