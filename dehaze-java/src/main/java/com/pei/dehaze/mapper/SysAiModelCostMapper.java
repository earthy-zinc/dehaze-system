package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysAiModelCost;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

@Mapper
public interface SysAiModelCostMapper extends BaseMapper<SysAiModelCost> {

    /**
     * 同模型同供应商的价格版本号递增。
     *
     * <p>版本号按全部历史（含软删）递增：联合唯一键含 deleted，软删版本号不可复用，
     * 故此处刻意不过滤 deleted（与 python include_deleted 口径一致）。
     */
    @Select("SELECT COALESCE(MAX(price_version), 0) + 1 FROM sys_ai_model_cost " +
            "WHERE model_id = #{modelId} AND provider_id = #{providerId}")
    int nextPriceVersion(@Param("modelId") String modelId, @Param("providerId") Long providerId);
}
