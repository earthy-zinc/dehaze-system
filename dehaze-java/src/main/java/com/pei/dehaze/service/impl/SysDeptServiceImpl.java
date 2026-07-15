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
import com.pei.dehaze.model.form.DeptForm;
import com.pei.dehaze.model.query.DeptQuery;
import com.pei.dehaze.model.vo.DeptVO;
import com.pei.dehaze.service.SysDeptService;
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

        // 生成部门路径(tree_path)，格式：父节点tree_path + , + 父节点ID，用于删除部门时级联删除子部门
        String treePath = generateDeptTreePath(parentId);
        entity.setTreePath(treePath);

        // 保存部门并返回部门ID
        boolean result = this.updateById(entity);
        Assert.isTrue(result, "部门更新失败");

        return entity.getId();
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
        // 批量删除部门及子部门，每个ID均参与级联匹配
        LambdaQueryWrapper<SysDept> wrapper = new LambdaQueryWrapper<SysDept>()
                .in(SysDept::getId, ids);
        for (Long id : ids) {
            wrapper.or().apply("CONCAT (',',tree_path,',') LIKE CONCAT('%,',{0},',%')", id);
        }
        boolean removed = this.remove(wrapper);
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
