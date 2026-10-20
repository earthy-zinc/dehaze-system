package com.pei.clarity.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.pei.clarity.mapper.SysAiUserPreferenceMapper;
import com.pei.clarity.model.entity.SysAiUserPreference;
import com.pei.clarity.model.form.AiUserPreferenceForm;
import com.pei.clarity.model.vo.AiUserPreferenceVO;
import com.pei.clarity.service.AiDefaultModelResolver;
import com.pei.clarity.service.AiUserPreferenceService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class AiUserPreferenceServiceImpl implements AiUserPreferenceService {

    private final SysAiUserPreferenceMapper preferenceMapper;

    private final AiDefaultModelResolver defaultModelResolver;

    @Override
    public AiUserPreferenceVO get(Long userId) {
        SysAiUserPreference preference = preferenceMapper.selectOne(
                new LambdaQueryWrapper<SysAiUserPreference>().eq(SysAiUserPreference::getUserId, userId));
        AiUserPreferenceVO vo = new AiUserPreferenceVO();
        vo.setDefaultModel(preference == null ? null : preference.getDefaultModel());
        return vo;
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void update(Long userId, AiUserPreferenceForm form) {
        String defaultModel = form.getDefaultModel();
        if (defaultModel != null) {
            defaultModelResolver.assertSelectable(defaultModel, userId);
        }
        preferenceMapper.upsertDefaultModel(userId, defaultModel);
    }
}
