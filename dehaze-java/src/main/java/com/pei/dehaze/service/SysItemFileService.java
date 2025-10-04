package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.bo.DatasetItemBO;
import com.pei.dehaze.model.dto.ImageFileInfo;
import com.pei.dehaze.model.entity.SysItemFile;
import com.pei.dehaze.model.vo.ImageUrlVO;

import java.util.List;

public interface SysItemFileService extends IService<SysItemFile> {
    ImageFileInfo saveItemFile(Long itemId, DatasetItemBO itemBO);

    List<ImageUrlVO> getImageUrlVOs(Long itemId);

    boolean deleteItemFile(Long itemId);
}
