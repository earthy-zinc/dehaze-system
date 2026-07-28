package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysMessage;
import org.apache.ibatis.annotations.Delete;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

@Mapper
public interface SysMessageMapper extends BaseMapper<SysMessage> {

    @Delete({
            "<script>",
            "DELETE FROM sys_message WHERE id IN",
            "<foreach collection='ids' item='id' open='(' separator=',' close=')'>#{id}</foreach>",
            "</script>"
    })
    int physicalDeleteByIds(@Param("ids") List<Long> ids);
}
