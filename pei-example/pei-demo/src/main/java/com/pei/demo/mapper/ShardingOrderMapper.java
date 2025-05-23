package com.pei.demo.mapper;

import com.baomidou.dynamic.datasource.annotation.DS;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;

import org.apache.ibatis.annotations.Mapper;
import com.pei.demo.domain.ShardingOrder;


@Mapper
@DS("sharding")
public interface ShardingOrderMapper extends BaseMapper<ShardingOrder> {


}
