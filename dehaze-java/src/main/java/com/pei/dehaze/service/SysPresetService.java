package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.entity.SysPreset;
import com.pei.dehaze.model.form.PresetForm;
import com.pei.dehaze.model.vo.PresetVO;

public interface SysPresetService extends IService<SysPreset> {

    /**
     * 获取预设列表（支持分页、算法筛选、系统/自定义过滤）
     */
    Page<PresetVO> listPresets(Long algorithmId, int pageNum, int pageSize, Boolean isSystem);

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
