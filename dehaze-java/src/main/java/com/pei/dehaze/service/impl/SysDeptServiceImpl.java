package com.pei.dehaze.service.impl;

import cn.hutool.core.collection.CollectionUtil;
import cn.hutool.core.lang.Assert;
import cn.hutool.core.util.StrUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.constant.SystemConstants;
import com.pei.dehaze.common.enums.StatusEnum;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.model.Option;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.common.util.TreeDataUtils;
import com.pei.dehaze.converter.DeptConverter;
import com.pei.dehaze.mapper.SysDeptMapper;
import com.pei.dehaze.model.entity.SysDept;
import com.pei.dehaze.model.entity.SysUser;
import com.pei.dehaze.model.form.DeptForm;
import com.pei.dehaze.model.query.DeptQuery;
import com.pei.dehaze.model.vo.DeptVO;
import com.pei.dehaze.service.SysDeptService;
import com.pei.dehaze.service.SysUserService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 部门业务实现类
 *
 * @author earthyzinc
 * @since 2021-08-22
 */
@Service
@RequiredArgsConstructor
public class SysDeptServiceImpl extends ServiceImpl<SysDeptMapper, SysDept> implements SysDeptService {


    private final DeptConverter deptConverter;
    private final SysUserService userService;

    /**
     * 部门最大层级深度（T-DPT-014/018a：超出 5 级报 A0504）
     */
    private static final int MAX_DEPT_LEVEL = 5;

    /**
     * 根部门ID（T-DPT-031/A0234：根部门不可删除）
     */
    private static final Long ROOT_DEPT_ID = 1L;

    /**
     * 获取部门列表
     */
    @Override
    public List<DeptVO> listDepartments(DeptQuery queryParams) {
        // 查询参数
        String keywords = queryParams.getKeywords();
        Integer status = queryParams.getStatus();

        // 查询数据
        List<SysDept> deptList = this.list(
                new LambdaQueryWrapper<SysDept>()
                        .like(StrUtil.isNotBlank(keywords), SysDept::getName, keywords)
                        .eq(status != null, SysDept::getStatus, status)
                        .orderByAsc(SysDept::getSort)
        );

        if (CollectionUtil.isEmpty(deptList)) {
            return Collections.emptyList();
        }

        // 获取根节点ID（父节点ID不在部门ID集合中的节点）
        List<Long> rootIds = TreeDataUtils.findRootIds(deptList, SysDept::getId, SysDept::getParentId);

        // 构建 parentId -> children Map，避免递归内 O(n) 过滤
        Map<Long, List<SysDept>> parentToChildrenMap = deptList.stream()
                .collect(Collectors.groupingBy(SysDept::getParentId));

        // 递归生成部门树形列表
        return rootIds.stream()
                .flatMap(rootId -> recurDeptList(rootId, parentToChildrenMap).stream())
                .toList();
    }

    /**
     * 递归生成部门树形列表
     *
     * @param parentId           父ID
     * @param parentToChildrenMap 父级ID -> 子部门列表 的Map（预先分组，O(1)查找）
     * @return 部门树形列表
     */
    private List<DeptVO> recurDeptList(Long parentId, Map<Long, List<SysDept>> parentToChildrenMap) {
        List<DeptVO> deptVOList = new ArrayList<>();
        List<SysDept> children = parentToChildrenMap.getOrDefault(parentId, Collections.emptyList());
        for (SysDept dept : children) {
            DeptVO deptVO = deptConverter.entity2Vo(dept);
            List<DeptVO> subList = recurDeptList(dept.getId(), parentToChildrenMap);
            if (!subList.isEmpty()) {
                deptVO.setChildren(subList);
            }
            deptVOList.add(deptVO);
        }
        return deptVOList;
    }

    /**
     * 部门下拉选项
     *
     * @return 部门下拉List集合
     */
    @Override
    public List<Option<Long>> listDeptOptions() {

        List<SysDept> deptList = this.list(new LambdaQueryWrapper<SysDept>()
                .eq(SysDept::getStatus, StatusEnum.ENABLE.getValue())
                .select(SysDept::getId, SysDept::getParentId, SysDept::getName)
                .orderByAsc(SysDept::getSort)
        );
        if (CollectionUtil.isEmpty(deptList)) {
            return Collections.emptyList();
        }

        List<Long> rootIds = TreeDataUtils.findRootIds(deptList, SysDept::getId, SysDept::getParentId);

        Map<Long, List<SysDept>> parentToChildrenMap = deptList.stream()
                .collect(Collectors.groupingBy(SysDept::getParentId));

        // 递归生成部门树形列表
        return rootIds.stream()
                .flatMap(rootId -> recurDeptTreeOptions(rootId, parentToChildrenMap).stream())
                .toList();
    }

    /**
     * 新增部门
     *
     * @param formData 部门表单
     * @return 部门ID
     */
    @Override
    public Long saveDept(DeptForm formData) {
        // 校验部门名称是否存在
        String name = formData.getName();
        long count = this.count(new LambdaQueryWrapper<SysDept>()
                .eq(SysDept::getName, name)
        );
        Assert.isTrue(count == 0, "部门名称已存在");

        // 生成部门路径(tree_path)，generateDeptTreePath 会校验父部门是否存在
        Long parentId = formData.getParentId();
        String treePath = generateDeptTreePath(parentId);
        assertMaxDeptDepth(treePath);

        // form->entity
        SysDept entity = deptConverter.form2Entity(formData);
        entity.setTreePath(treePath);

        // 保存部门并返回部门ID
        boolean result = this.save(entity);
        Assert.isTrue(result, "部门保存失败");

        return entity.getId();
    }

    /**
     * 更新部门
     *
     * @param deptId   部门ID
     * @param formData 部门表单
     * @return 部门ID
     */
    @Override
    public Long updateDept(Long deptId, DeptForm formData) {
        // 校验部门名称是否存在
        String name = formData.getName();
        long count = this.count(new LambdaQueryWrapper<SysDept>()
                .eq(SysDept::getName, name)
                .ne(SysDept::getId, deptId)
        );
        Assert.isTrue(count == 0, "部门名称已存在");

        // 循环引用校验：不能将部门移动到自身或其子部门下
        Long parentId = formData.getParentId();
        Assert.isTrue(!parentId.equals(deptId), "不能将部门设置为自己的上级部门");
        if (!SystemConstants.ROOT_NODE_ID.equals(parentId)) {
            SysDept parentDept = this.getById(parentId);
            Assert.isTrue(parentDept != null, "父部门不存在");
            // 父部门的 tree_path 包含当前部门ID → 父部门是当前部门的子部门 → 循环引用
            if (parentDept.getTreePath() != null) {
                String treePathWithCommas = "," + parentDept.getTreePath() + ",";
                Assert.isTrue(!treePathWithCommas.contains("," + deptId + ","),
                        "不能将部门移动到其子部门下，存在循环引用");
            }
        }

        // form->entity
        SysDept entity = deptConverter.form2Entity(formData);
        entity.setId(deptId);

        // 生成部门路径(tree_path)，格式：父节点tree_path + , + 父节点ID
        String treePath = generateDeptTreePath(parentId);
        assertMaxDeptDepth(treePath);
        entity.setTreePath(treePath);

        // 保存部门并返回部门ID
        boolean result = this.updateById(entity);
        Assert.isTrue(result, "部门更新失败");

        return entity.getId();
    }

    /**
     * 校验部门层级不超过 5 级（T-DPT-014/018a，超限返回 A0504）
     *
     * @param treePath 部门路径，逗号分隔的父节点ID链
     */
    private void assertMaxDeptDepth(String treePath) {
        if (StrUtil.isBlank(treePath)) {
            return;
        }
        int depth = StrUtil.split(treePath, ',').size();
        if (depth > MAX_DEPT_LEVEL) {
            throw new BusinessException(ResultCode.DATA_BIND_EXISTS, "部门层级不能超过5级");
        }
    }

    /**
     * 递归生成部门表格层级列表
     *
     * @param parentId           父ID
     * @param parentToChildrenMap 父级ID -> 子部门列表 的Map（预先分组，O(1)查找）
     * @return 部门表格层级列表
     */
    private List<Option<Long>> recurDeptTreeOptions(Long parentId, Map<Long, List<SysDept>> parentToChildrenMap) {
        List<Option<Long>> optionList = new ArrayList<>();
        List<SysDept> children = parentToChildrenMap.getOrDefault(parentId, Collections.emptyList());
        for (SysDept dept : children) {
            Option<Long> option = new Option<>(dept.getId(), dept.getName());
            List<Option<Long>> subOptions = recurDeptTreeOptions(dept.getId(), parentToChildrenMap);
            if (!subOptions.isEmpty()) {
                option.setChildren(subOptions);
            }
            optionList.add(option);
        }
        return optionList;
    }


    /**
     * 删除部门
     *
     * @param ids 部门ID列表
     * @return 是否删除成功
     */
    @Override
    @Transactional(rollbackFor = Exception.class)
    public boolean deleteByIds(List<Long> ids) {
        if (CollectionUtil.isEmpty(ids)) {
            return true;
        }
        // 批量前置校验：全部通过后统一逻辑删除，任一不满足则整体失败（T-DPT-028/029/030/031）
        for (Long id : ids) {
            SysDept dept = this.getById(id);
            if (dept == null) {
                throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "部门不存在");
            }
            // 根部门保护（T-DPT-031，返回 A0234）
            if (ROOT_DEPT_ID.equals(id)) {
                throw new BusinessException(ResultCode.OPERATION_NOT_ALLOW, "根部门不可删除");
            }
            // 子部门检查：有子部门禁止删除（T-DPT-030，不级联删除）
            long childCount = this.count(new LambdaQueryWrapper<SysDept>()
                    .eq(SysDept::getParentId, id));
            if (childCount > 0) {
                throw new BusinessException(ResultCode.DATA_BIND_EXISTS,
                        "该部门下存在子部门，请先删除子部门");
            }
            // 关联用户检查：有用户禁止删除（T-DPT-029）
            long userCount = userService.count(new LambdaQueryWrapper<SysUser>()
                    .eq(SysUser::getDeptId, id));
            if (userCount > 0) {
                throw new BusinessException(ResultCode.DATA_BIND_EXISTS,
                        "该部门下存在用户，无法删除");
            }
        }

        // 全部通过后逻辑删除
        boolean removed = this.removeByIds(ids);
        if (!removed) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "部门不存在");
        }
        return true;
    }

    /**
     * 获取部门详情
     *
     * @param deptId 部门ID
     * @return 部门表单对象
     */
    @Override
    public DeptForm getDeptForm(Long deptId) {

        SysDept entity = this.getOne(new LambdaQueryWrapper<SysDept>()
                .eq(SysDept::getId, deptId)
                .select(
                        SysDept::getId,
                        SysDept::getName,
                        SysDept::getParentId,
                        SysDept::getStatus,
                        SysDept::getSort
                ));
        if (entity == null) {
            return null;
        }
        return deptConverter.entity2Form(entity);
    }


    /**
     * 部门路径生成
     *
     * @param parentId 父ID
     * @return 父节点路径以英文逗号(, )分割，eg: 1,2,3
     */
    private String generateDeptTreePath(Long parentId) {
        if (SystemConstants.ROOT_NODE_ID.equals(parentId)) {
            return String.valueOf(parentId);
        }
        SysDept parent = this.getById(parentId);
        if (parent == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "父部门不存在");
        }
        return parent.getTreePath() + "," + parent.getId();
    }
}
