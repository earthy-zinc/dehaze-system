package com.pei.clarity.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.clarity.common.model.Option;
import com.pei.clarity.model.entity.SysDept;
import com.pei.clarity.model.form.DeptForm;
import com.pei.clarity.model.query.DeptQuery;
import com.pei.clarity.model.vo.DeptVO;
import java.util.List;

/**
 * 部门业务接口
 *
 * @author earthyzinc
 * @since 2021/8/22
 */
public interface SysDeptService extends IService<SysDept> {
    /**
     * 部门列表
     *
     * @return
     */
    List<DeptVO> listDepartments(DeptQuery queryParams);

    /**
     * 部门树形下拉选项
     *
     * @return
     */
    List<Option<Long>> listDeptOptions();

    /**
     * 新增部门
     *
     * @param formData
     * @return
     */
    Long saveDept(DeptForm formData);

    /**
     * 修改部门
     *
     * @param deptId
     * @param formData
     * @return
     */
    Long updateDept(Long deptId, DeptForm formData);

    /**
     * 删除部门
     *
     * @param ids 部门ID列表
     * @return 是否删除成功
     */
    boolean deleteByIds(List<Long> ids);

    /**
     * 获取部门详情
     *
     * @param deptId
     * @return
     */
    DeptForm getDeptForm(Long deptId);
}
