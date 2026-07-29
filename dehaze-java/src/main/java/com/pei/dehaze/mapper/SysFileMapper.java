package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysFile;
import org.apache.ibatis.annotations.Delete;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

@Mapper
public interface SysFileMapper extends BaseMapper<SysFile> {

    @Select("SELECT * FROM sys_file WHERE md5 = #{md5} LIMIT 1")
    SysFile selectByMd5IncludeDeleted(String md5);

    @Delete("DELETE FROM sys_file WHERE id = #{id}")
    int hardDeleteById(Long id);
}
