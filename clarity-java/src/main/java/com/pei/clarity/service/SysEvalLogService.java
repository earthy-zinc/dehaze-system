package com.pei.clarity.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.clarity.model.entity.SysEvalLog;
import com.pei.clarity.model.form.EvaluationForm;
import com.pei.clarity.model.query.EvalLogQuery;
import com.pei.clarity.model.vo.EvalLogVO;
import com.pei.clarity.model.vo.EvaluationResultVO;

public interface SysEvalLogService extends IService<SysEvalLog> {

    /**
     * 执行效果评估
     */
    EvaluationResultVO evaluate(EvaluationForm form);

    /**
     * 获取指定用户的评估日志分页列表
     */
    Page<EvalLogVO> getEvalLogPage(EvalLogQuery query, Long userId);
}
