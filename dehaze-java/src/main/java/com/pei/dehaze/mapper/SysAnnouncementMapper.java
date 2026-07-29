package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysAnnouncement;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface SysAnnouncementMapper extends BaseMapper<SysAnnouncement> {

    @Select("SELECT user_id FROM sys_member WHERE level_code = CONCAT('level_', #{level}) AND deleted = 0")
    List<Long> selectUserIdsByLevel(@Param("level") Integer level);

    @Select("SELECT id FROM sys_user WHERE deleted = 0 AND status = 1 AND dept_id IN (SELECT id FROM sys_dept WHERE deleted = 0 AND status = 1 AND name = #{tag})")
    List<Long> selectUserIdsByTag(@Param("tag") String tag);
}
