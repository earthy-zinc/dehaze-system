package com.pei.clarity.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.clarity.model.entity.SysMessageTemplate;
import com.pei.clarity.model.form.MessageTemplateForm;
import com.pei.clarity.model.query.MessageTemplateQuery;
import com.pei.clarity.model.vo.MessageTemplateDetailVO;
import com.pei.clarity.model.vo.MessageTemplateVO;

public interface MessageTemplateService extends IService<SysMessageTemplate> {

    Page<MessageTemplateVO> getPage(MessageTemplateQuery query);

    MessageTemplateDetailVO getDetail(Long id);

    void update(Long id, MessageTemplateForm form);

    SysMessageTemplate getByCode(String code);
}
