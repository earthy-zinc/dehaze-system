package com.pei.dehaze.module.ai.dal.mysql.mindmap;

import com.pei.dehaze.framework.common.pojo.PageResult;
import com.pei.dehaze.framework.mybatis.core.mapper.BaseMapperX;
import com.pei.dehaze.framework.mybatis.core.query.LambdaQueryWrapperX;
import com.pei.dehaze.module.ai.controller.admin.mindmap.vo.AiMindMapPageReqVO;
import com.pei.dehaze.module.ai.dal.dataobject.mindmap.AiMindMapDO;
import org.apache.ibatis.annotations.Mapper;

/**
 * AI 思维导图 Mapper
 *
 * @author xiaoxin
 */
@Mapper
public interface AiMindMapMapper extends BaseMapperX<AiMindMapDO> {

    default PageResult<AiMindMapDO> selectPage(AiMindMapPageReqVO reqVO) {
        return selectPage(reqVO, new LambdaQueryWrapperX<AiMindMapDO>()
                .eqIfPresent(AiMindMapDO::getUserId, reqVO.getUserId())
                .eqIfPresent(AiMindMapDO::getPrompt, reqVO.getPrompt())
                .betweenIfPresent(AiMindMapDO::getCreateTime, reqVO.getCreateTime())
                .orderByDesc(AiMindMapDO::getId));
    }

}
