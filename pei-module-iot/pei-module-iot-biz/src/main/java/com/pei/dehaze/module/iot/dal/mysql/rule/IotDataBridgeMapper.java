package com.pei.dehaze.module.iot.dal.mysql.rule;

import com.pei.dehaze.framework.common.pojo.PageResult;
import com.pei.dehaze.framework.mybatis.core.mapper.BaseMapperX;
import com.pei.dehaze.framework.mybatis.core.query.LambdaQueryWrapperX;
import com.pei.dehaze.module.iot.controller.admin.rule.vo.databridge.IotDataBridgePageReqVO;
import com.pei.dehaze.module.iot.dal.dataobject.rule.IotDataBridgeDO;
import org.apache.ibatis.annotations.Mapper;

/**
 * IoT 数据桥梁 Mapper
 *
 * @author HUIHUI
 */
@Mapper
public interface IotDataBridgeMapper extends BaseMapperX<IotDataBridgeDO> {

    default PageResult<IotDataBridgeDO> selectPage(IotDataBridgePageReqVO reqVO) {
        return selectPage(reqVO, new LambdaQueryWrapperX<IotDataBridgeDO>()
                .likeIfPresent(IotDataBridgeDO::getName, reqVO.getName())
                .eqIfPresent(IotDataBridgeDO::getStatus, reqVO.getStatus())
                .betweenIfPresent(IotDataBridgeDO::getCreateTime, reqVO.getCreateTime())
                .orderByDesc(IotDataBridgeDO::getId));
    }

}
