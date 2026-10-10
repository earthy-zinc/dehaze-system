package com.pei.dehaze.service;

import com.pei.dehaze.model.form.AiUserPreferenceForm;
import com.pei.dehaze.model.vo.AiUserPreferenceVO;

public interface AiUserPreferenceService {

    /** 用户偏好默认模型；无记录或空值返回 null（不自动建行） */
    AiUserPreferenceVO get(Long userId);

    /** 写入偏好默认模型；defaultModel 为 null 即清除偏好 */
    void update(Long userId, AiUserPreferenceForm form);
}
