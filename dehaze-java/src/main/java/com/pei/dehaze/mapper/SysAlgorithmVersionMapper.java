package com.pei.dehaze.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.dehaze.model.entity.SysAlgorithmVersion;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

/**
 * 算法版本访问层，提供绕过@TableLogic软删过滤的原生SQL查询方法。
 */
@Mapper
public interface SysAlgorithmVersionMapper extends BaseMapper<SysAlgorithmVersion> {

    /**
     * 按算法ID+版本号查询算法版本数（含软删行，绕过@TableLogic过滤）
     *
     * @param algorithmId 算法ID
     * @param version 版本号
     * @return 匹配记录数
     */
    long countByAlgorithmIdAndVersionAll(
        @Param("algorithmId") Long algorithmId,
        @Param("version") String version
    );

    /**
     * 按算法ID+版本号查询算法版本数（排除指定ID，含软删行）
     *
     * @param algorithmId 算法ID
     * @param version 版本号
     * @param excludeId 排除的ID
     * @return 匹配记录数
     */
    long countByAlgorithmIdAndVersionAllExcluding(
        @Param("algorithmId") Long algorithmId,
        @Param("version") String version,
        @Param("excludeId") Long excludeId
    );
}
