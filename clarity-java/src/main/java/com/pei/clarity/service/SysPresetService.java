package com.pei.clarity.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.clarity.model.entity.SysPreset;
import com.pei.clarity.model.form.PresetForm;
import com.pei.clarity.model.vo.PresetVO;

public interface SysPresetService extends IService<SysPreset> {

    /**
     * 获取预设列表（支持分页、算法筛选、系统/自定义过滤）
     */
    Page<PresetVO> listPresets(String modelId, int pageNum, int pageSize, Boolean isSystem);

    /**
     * 创建自定义预设
     */
    PresetVO createPreset(PresetForm form);

    /**
     * 更新自定义预设
     */
    PresetVO updatePreset(Long id, PresetForm form);

    /**
     * 删除自定义预设
     */
    void deletePreset(Long id);
}
