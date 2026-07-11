package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.entity.SysEvalLog;
import com.pei.dehaze.model.form.EvaluationForm;
import com.pei.dehaze.model.query.EvalLogQuery;
import com.pei.dehaze.model.vo.EvalLogVO;
import com.pei.dehaze.model.vo.EvaluationResultVO;

public interface SysEvalLogService extends IService<SysEvalLog> {

    /**
     * 执行效果评估
     */
    EvaluationResultVO evaluate(EvaluationForm form);

    /**
     * 获取评估日志分页列表
     */
    Page<EvalLogVO> getEvalLogPage(EvalLogQuery query);
}
