package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.entity.SysAlgorithmVersion;
import com.pei.dehaze.model.form.AlgorithmVersionForm;
import com.pei.dehaze.model.vo.AlgorithmVersionVO;

import java.util.List;

/**
 * @author earthyzinc
 * @since 2024-06-12
 */
public interface SysAlgorithmVersionService extends IService<SysAlgorithmVersion> {

    /**
     * 获取算法版本历史列表
     */
    List<AlgorithmVersionVO> getVersionHistory(Long algorithmId);

    /**
     * 新增算法版本
     */
    SysAlgorithmVersion addVersion(Long algorithmId, AlgorithmVersionForm form);

    /**
     * 回滚到指定版本
     */
    void rollbackToVersion(Long algorithmId, Long versionId);
}
