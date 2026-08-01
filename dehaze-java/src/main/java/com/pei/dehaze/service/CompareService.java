package com.pei.dehaze.service;

import com.pei.dehaze.model.form.CompareReportForm;
import com.pei.dehaze.model.vo.CompareReportResultVO;

/**
 * 效果对比服务接口
 */
public interface CompareService {

    /**
     * 生成对比报告（异步任务）
     *
     * @param form 报告生成表单
     * @return 任务结果
     */
    CompareReportResultVO generateReport(CompareReportForm form);

    /**
     * 获取对比报告下载URL
     *
     * @param taskId 任务ID
     * @return 下载URL
     */
    String getReportDownloadUrl(Long taskId);
}
