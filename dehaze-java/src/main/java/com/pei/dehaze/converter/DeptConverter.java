package com.pei.dehaze.converter;

import com.pei.dehaze.model.entity.SysDept;
import com.pei.dehaze.model.form.DeptForm;
import com.pei.dehaze.model.vo.DeptVO;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;
import org.mapstruct.Mappings;

/**
 * 部门对象转换器
 *
 * @author earthyzinc
 * @since 2022/7/29
 */
@Mapper(componentModel = "spring")
public interface DeptConverter {

    DeptForm entity2Form(SysDept entity);

    @Mapping(ignore = true, target = "children")
    DeptVO entity2Vo(SysDept entity);

    @Mappings({
        @Mapping(target = "createTime", ignore = true),
        @Mapping(target = "updateTime", ignore = true),
        @Mapping(target = "createBy", ignore = true),
        @Mapping(target = "deleted", ignore = true),
        @Mapping(target = "treePath", ignore = true),
        @Mapping(target = "updateBy", ignore = true)
    })
    SysDept form2Entity(DeptForm deptForm);

}
