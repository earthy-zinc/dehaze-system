package com.pei.dehaze.service.impl;

import cn.hutool.core.text.CharSequenceUtil;
import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.mapper.SysMessageTemplateMapper;
import com.pei.dehaze.model.entity.SysMessageTemplate;
import com.pei.dehaze.model.form.MessageTemplateForm;
import com.pei.dehaze.model.query.MessageTemplateQuery;
import com.pei.dehaze.model.vo.MessageTemplateDetailVO;
import com.pei.dehaze.model.vo.MessageTemplateVO;
import com.pei.dehaze.service.MessageTemplateService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;

@Service
@RequiredArgsConstructor
public class MessageTemplateServiceImpl extends ServiceImpl<SysMessageTemplateMapper, SysMessageTemplate> implements MessageTemplateService {

    @Override
    public Page<MessageTemplateVO> getPage(MessageTemplateQuery query) {
        Page<SysMessageTemplate> page = new Page<>(query.getPageNum(), query.getPageSize());
        this.page(page, new LambdaQueryWrapper<SysMessageTemplate>()
                .like(CharSequenceUtil.isNotBlank(query.getName()), SysMessageTemplate::getName, query.getName())
                .eq(CharSequenceUtil.isNotBlank(query.getType()), SysMessageTemplate::getType, query.getType())
                .eq(query.getStatus() != null, SysMessageTemplate::getStatus, query.getStatus())
                .orderByDesc(SysMessageTemplate::getCreateTime));

        Page<MessageTemplateVO> result = new Page<>(page.getCurrent(), page.getSize(), page.getTotal());
        result.setRecords(page.getRecords().stream().map(this::toTemplateVO).toList());
        return result;
    }

    @Override
    public MessageTemplateDetailVO getDetail(Long id) {
        SysMessageTemplate template = this.getById(id);
        if (template == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND);
        }
        MessageTemplateDetailVO vo = new MessageTemplateDetailVO();
        vo.setId(template.getId());
        vo.setCode(template.getCode());
        vo.setName(template.getName());
        vo.setType(template.getType());
        vo.setTitleTemplate(template.getTitleTemplate());
        vo.setContentTemplate(template.getContentTemplate());
        vo.setPriority(template.getPriority());
        vo.setStatus(template.getStatus());
        vo.setCreateTime(template.getCreateTime());
        vo.setUpdateTime(template.getUpdateTime());
        if (CharSequenceUtil.isNotBlank(template.getChannels())) {
            vo.setChannels(JSONUtil.parseObj(template.getChannels()).toBean(Map.class));
        }
        if (CharSequenceUtil.isNotBlank(template.getVariables())) {
            @SuppressWarnings({"unchecked", "rawtypes"})
            List<Map<String, String>> vars = (List) JSONUtil.parseArray(template.getVariables()).toList(Map.class);
            vo.setVariables(vars);
        }
        return vo;
    }

    @Override
    public void update(Long id, MessageTemplateForm form) {
        SysMessageTemplate template = this.getById(id);
        if (template == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND);
        }
        SysMessageTemplate entity = new SysMessageTemplate();
        entity.setId(id);
        if (form.getName() != null) {
            entity.setName(form.getName());
        }
        if (form.getTitleTemplate() != null) {
            entity.setTitleTemplate(form.getTitleTemplate());
        }
        if (form.getContentTemplate() != null) {
            entity.setContentTemplate(form.getContentTemplate());
        }
        if (form.getPriority() != null) {
            entity.setPriority(form.getPriority());
        }
        if (form.getChannels() != null) {
            entity.setChannels(JSONUtil.toJsonStr(form.getChannels()));
        }
        if (form.getStatus() != null) {
            entity.setStatus(form.getStatus());
        }
        this.updateById(entity);
    }

    @Override
    public SysMessageTemplate getByCode(String code) {
        return this.getOne(new LambdaQueryWrapper<SysMessageTemplate>()
                .eq(SysMessageTemplate::getCode, code));
    }

    private MessageTemplateVO toTemplateVO(SysMessageTemplate template) {
        MessageTemplateVO vo = new MessageTemplateVO();
        vo.setId(template.getId());
        vo.setCode(template.getCode());
        vo.setName(template.getName());
        vo.setType(template.getType());
        vo.setTitleTemplate(template.getTitleTemplate());
        vo.setPriority(template.getPriority());
        vo.setStatus(template.getStatus());
        vo.setCreateTime(template.getCreateTime());
        return vo;
    }
}
