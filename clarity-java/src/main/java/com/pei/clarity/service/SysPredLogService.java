package com.pei.clarity.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.clarity.model.entity.SysPredLog;
import com.pei.clarity.model.form.PredictionForm;
import com.pei.clarity.model.query.PredLogQuery;
import com.pei.clarity.model.vo.PredLogVO;
import com.pei.clarity.model.vo.PredictionResultVO;

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
