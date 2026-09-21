package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.conditions.Wrapper;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.toolkit.Constants;
import com.pei.dehaze.model.entity.SysDept;
import com.pei.dehaze.plugin.mybatis.annotation.DataPermission;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.util.List;


@Mapper
public interface SysDeptMapper extends BaseMapper<SysDept> {

    @DataPermission(deptIdColumnName = "id")
    @Override
    List<SysDept> selectList(@Param(Constants.WRAPPER) Wrapper<SysDept> queryWrapper);

    /**
     * 统计同级同名部门数（含软删行，绕过 @TableLogic 过滤）。
     * python check_name_exists include_deleted 口径：删除后同级名称不可复用（T-DPT-035b）。
     */
    @Select("<script>" +
            "SELECT COUNT(*) FROM sys_dept WHERE name = #{name} AND parent_id = #{parentId}" +
            "<if test='excludeId != null'> AND id != #{excludeId}</if>" +
            "</script>")
    long countByNameIncludingDeleted(@Param("name") String name, @Param("parentId") Long parentId,
                                     @Param("excludeId") Long excludeId);

    /**
     * 移动部门后级联平移子树 tree_path：旧前缀整体替换为新前缀（含软删行，保持路径一致性）
     *
     * @param prefixLen 旧前缀长度（SUBSTRING 为 1 起始位置，从此处截取后缀）
     */
    @Update("UPDATE sys_dept SET tree_path = CONCAT(#{newPrefix}, SUBSTRING(tree_path, #{prefixLen} + 1)) " +
            "WHERE tree_path = #{oldPrefix} OR tree_path LIKE CONCAT(#{oldPrefix}, ',%')")
    int updateSubtreeTreePath(@Param("oldPrefix") String oldPrefix,
                              @Param("newPrefix") String newPrefix,
                              @Param("prefixLen") int prefixLen);
}
