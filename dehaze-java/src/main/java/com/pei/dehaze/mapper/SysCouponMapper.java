package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysCoupon;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.Collection;
import java.util.List;

@Mapper
public interface SysCouponMapper extends BaseMapper<SysCoupon> {

    @Select({
        "<script>",
        "SELECT * FROM sys_coupon WHERE id IN",
        "<foreach collection='ids' item='id' open='(' separator=',' close=')'>#{id}</foreach>",
        "</script>"
    })
    List<SysCoupon> selectByIdsIncludeDeleted(@Param("ids") Collection<Long> ids);
}
