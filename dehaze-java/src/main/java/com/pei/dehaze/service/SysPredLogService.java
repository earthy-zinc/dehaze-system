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
     * 提交图像任务（异步）
     */
    PredictionResultVO predict(PredictionForm form);

    /**
     * 获取当前用户图像任务历史分页
     */
    Page<PredLogVO> getPredLogPage(PredLogQuery query);
}
