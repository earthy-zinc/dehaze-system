package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.entity.SysPredLog;
import com.pei.dehaze.model.form.PredictionForm;
import com.pei.dehaze.model.query.PredLogQuery;
import com.pei.dehaze.model.vo.PredLogVO;
import com.pei.dehaze.model.vo.PredictionResultVO;

public interface SysPredLogService extends IService<SysPredLog> {

    /**
     * 执行模型预测
     */
    PredictionResultVO predict(PredictionForm form);

    /**
     * 获取预测日志分页列表
     */
    Page<PredLogVO> getPredLogPage(PredLogQuery query);
}
