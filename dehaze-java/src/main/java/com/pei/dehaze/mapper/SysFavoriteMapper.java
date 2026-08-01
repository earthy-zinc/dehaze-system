package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.model.entity.SysFavorite;
import com.pei.dehaze.model.vo.FavoriteVO;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Options;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

@Mapper
public interface SysFavoriteMapper extends BaseMapper<SysFavorite> {

    /**
     * 收藏列表分页查询，SQL 层 LEFT JOIN 各业务表取对象名称/缩略图并做关键词过滤，保证 total 准确。
     * 算法关联 sys_algorithm，数据集关联 sys_dataset；其余类型 targetName 为 null。
     * JOIN 条件带 deleted=0，对象被逻辑删除后 JOIN 不上，配合 is_invalid 字段标记失效。
     */
    @Select("<script>" +
            "SELECT f.id, f.user_id, f.target_type, f.target_id, f.is_invalid, f.create_time, " +
            "CASE f.target_type " +
            "  WHEN 'algorithm' THEN a.name " +
            "  WHEN 'dataset' THEN d.name " +
            "  ELSE NULL " +
            "END AS target_name, " +
            "CASE f.target_type " +
            "  WHEN 'dataset' THEN d.img " +
            "  ELSE NULL " +
            "END AS target_thumbnail " +
            "FROM sys_favorite f " +
            "LEFT JOIN sys_algorithm a ON f.target_type = 'algorithm' AND f.target_id = a.id AND a.deleted = 0 " +
            "LEFT JOIN sys_dataset d ON f.target_type = 'dataset' AND f.target_id = d.id AND d.deleted = 0 " +
            "WHERE f.deleted = 0 AND f.user_id = #{userId} " +
            "<if test='targetType != null and targetType != \"\"'> AND f.target_type = #{targetType}</if> " +
            "<if test='keywords != null and keywords != \"\"'>" +
            " AND CASE f.target_type WHEN 'algorithm' THEN a.name WHEN 'dataset' THEN d.name ELSE NULL END LIKE CONCAT('%', #{keywords}, '%')" +
            "</if> " +
            "ORDER BY " +
            "<choose>" +
            "  <when test='sortBy == \"createTime\" and sortOrder == \"asc\"'>f.create_time ASC</when>" +
            "  <otherwise>f.create_time DESC</otherwise>" +
            "</choose>" +
            "</script>")
    Page<FavoriteVO> selectFavoritePage(Page<FavoriteVO> page,
                                        @Param("userId") Long userId,
                                        @Param("targetType") String targetType,
                                        @Param("keywords") String keywords,
                                        @Param("sortBy") String sortBy,
                                        @Param("sortOrder") String sortOrder);

    /**
     * upsert 收藏：user_id + target_type + target_id 唯一键冲突时复活软删行。
     * UPDATE 分支通过 LAST_INSERT_ID(id) 拿回原行 id，MyBatis 回填到 entity.id。
     */
    @Insert("INSERT INTO sys_favorite (user_id, target_type, target_id, is_invalid, deleted, update_time) " +
            "VALUES (#{userId}, #{targetType}, #{targetId}, 0, 0, NOW()) " +
            "ON DUPLICATE KEY UPDATE id = LAST_INSERT_ID(id), deleted = 0, is_invalid = 0, update_time = NOW()")
    @Options(useGeneratedKeys = true, keyProperty = "id")
    int upsertByUserAndTarget(SysFavorite favorite);
}
