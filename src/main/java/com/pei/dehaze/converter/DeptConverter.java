package com.pei.dehaze.converter;

import com.pei.dehaze.model.entity.SysDept;
import com.pei.dehaze.model.form.DeptForm;
import com.pei.dehaze.model.vo.DeptVO;
import org.mapstruct.Mapper;

/**
 * 部门对象转换器
 *
 * @author earthyzinc
 * @since 2022/7/29
 */
@Mapper(componentModel = "spring")
public interface DeptConverter {

    DeptForm entity2Form(SysDept entity);
    DeptVO entity2Vo(SysDept entity);

    SysDept form2Entity(DeptForm deptForm);

}
