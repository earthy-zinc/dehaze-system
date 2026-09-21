package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysAiModelPrice;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

@Mapper
public interface SysAiModelPriceMapper extends BaseMapper<SysAiModelPrice> {

    /**
     * 下一个价格版本号：按全部历史（含软删）递增，删除后不回退版本号。
     */
    @Select("SELECT COALESCE(MAX(price_version), 0) + 1 FROM sys_ai_model_price "
            + "WHERE model_id = #{modelId} AND provider_id = #{providerId}")
    int nextPriceVersion(@Param("modelId") String modelId, @Param("providerId") Long providerId);
}
