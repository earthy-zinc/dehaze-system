package com.pei.dehaze.module.system.service.logger;

import com.pei.dehaze.framework.common.pojo.PageResult;
import com.pei.dehaze.framework.common.util.object.BeanUtils;
import com.pei.dehaze.framework.common.biz.system.logger.dto.OperateLogCreateReqDTO;
import com.pei.dehaze.module.system.api.logger.dto.OperateLogPageReqDTO;
import com.pei.dehaze.module.system.controller.admin.logger.vo.operatelog.OperateLogPageReqVO;
import com.pei.dehaze.module.system.dal.dataobject.logger.OperateLogDO;
import com.pei.dehaze.module.system.dal.mysql.logger.OperateLogMapper;
import jakarta.annotation.Resource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.validation.annotation.Validated;

/**
 * 操作日志 Service 实现类
 *
 * @author earthyzinc
 */
@Service
@Validated
@Slf4j
public class OperateLogServiceImpl implements OperateLogService {

    @Resource
    private OperateLogMapper operateLogMapper;

    @Override
    public void createOperateLog(OperateLogCreateReqDTO createReqDTO) {
        OperateLogDO log = BeanUtils.toBean(createReqDTO, OperateLogDO.class);
        operateLogMapper.insert(log);
    }

    @Override
    public PageResult<OperateLogDO> getOperateLogPage(OperateLogPageReqVO pageReqVO) {
        return operateLogMapper.selectPage(pageReqVO);
    }

    @Override
    public PageResult<OperateLogDO> getOperateLogPage(OperateLogPageReqDTO pageReqDTO) {
        return operateLogMapper.selectPage(pageReqDTO);
    }

}
