package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysAiProvider;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

@Mapper
public interface SysAiProviderMapper extends BaseMapper<SysAiProvider> {

    /**
     * provider_code 占用行数（含软删行）。
     *
     * <p>provider_code 为白名单语义业务键：软删后不可复用，故查重必须绕过逻辑删除过滤。
     */
    @Select("SELECT COUNT(*) FROM sys_ai_provider WHERE provider_code = #{providerCode}")
    long countByProviderCodeIncludingDeleted(@Param("providerCode") String providerCode);
}
