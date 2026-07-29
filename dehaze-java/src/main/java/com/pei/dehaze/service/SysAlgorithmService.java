package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.common.model.Option;
import com.pei.dehaze.model.entity.SysAlgorithm;
import com.pei.dehaze.model.form.AlgorithmAuditForm;
import com.pei.dehaze.model.form.AlgorithmForm;
import com.pei.dehaze.model.query.AlgorithmQuery;
import com.pei.dehaze.model.vo.AlgorithmMonitorVO;
import com.pei.dehaze.model.vo.AlgorithmVO;

import java.util.List;
import java.util.Map;

/**
 * @author earthy-zinc
 * @since 2024-06-08 18:34:16
 */
public interface SysAlgorithmService extends IService<SysAlgorithm> {
    List<SysAlgorithm> getAllAlgorithms();

    List<AlgorithmVO> getList(AlgorithmQuery queryParams);

    List<Option<Long>> getOption();

    List<AlgorithmVO> listAll();

    SysAlgorithm getAlgorithmById(Long id);

    SysAlgorithm getRootAlgorithm(Long id);

    Long addAlgorithm(AlgorithmForm algorithm);

    boolean updateAlgorithm(AlgorithmForm algorithm);

    boolean deleteAlgorithms(List<Long> ids);

    /**
     * 修改算法状态
     */
    boolean updateStatus(Long id, Integer status);

    /**
     * 审核算法（通过/驳回）
     */
    boolean auditAlgorithm(Long id, AlgorithmAuditForm form);

    /**
     * 获取算法监控数据
     */
    AlgorithmMonitorVO getMonitorData(Long id);

    /**
     * 获取算法统计报表（按日聚合）
     * @param days 统计天数，默认7天
     */
    List<Map<String, Object>> getMonitorStats(Long id, Integer days);
}
