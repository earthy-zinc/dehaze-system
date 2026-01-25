package com.pei.dehaze.converter;

import com.pei.dehaze.common.enums.StatusEnum;
import com.pei.dehaze.model.dto.DatasetStatistics;
import com.pei.dehaze.model.entity.SysDataset;
import com.pei.dehaze.model.form.DatasetAddForm;
import com.pei.dehaze.model.form.DatasetUpdateForm;
import com.pei.dehaze.model.vo.DatasetVO;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;

/**
 * @author earthy-zinc
 * @since 2024-06-09 00:11:09
 */
@Mapper(componentModel = "spring")
public interface DatasetConverter {

    @Mapping(ignore = true, target = "children")
    @Mapping(ignore = true, target = "hasChildren")
    @Mapping(source = "entity.status", target = "status")
    DatasetVO entity2Vo(SysDataset entity, DatasetStatistics statistics);

    @Mapping(ignore = true, target = "id")
    @Mapping(ignore = true, target = "img")
    @Mapping(ignore = true, target = "path")
    @Mapping(ignore = true, target = "size")
    @Mapping(ignore = true, target = "deleted")
    @Mapping(ignore = true, target = "usageCount")
    @Mapping(ignore = true, target = "createBy")
    @Mapping(ignore = true, target = "updateBy")
    @Mapping(ignore = true, target = "createTime")
    @Mapping(ignore = true, target = "updateTime")
    SysDataset form2Entity(DatasetAddForm form);

    @Mapping(ignore = true, target = "parentId")
    @Mapping(ignore = true, target = "img")
    @Mapping(ignore = true, target = "path")
    @Mapping(ignore = true, target = "size")
    @Mapping(ignore = true, target = "deleted")
    @Mapping(ignore = true, target = "usageCount")
    @Mapping(ignore = true, target = "createBy")
    @Mapping(ignore = true, target = "updateBy")
    @Mapping(ignore = true, target = "createTime")
    @Mapping(ignore = true, target = "updateTime")
    SysDataset updateForm2Entity(DatasetUpdateForm vo);

    /**
     * StatusEnum转Integer
     */
    default Integer map(StatusEnum status) {
        return status == null ? null : status.getValue();
    }
}
