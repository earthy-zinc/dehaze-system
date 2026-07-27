package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.entity.SysMessageTemplate;
import com.pei.dehaze.model.form.MessageTemplateForm;
import com.pei.dehaze.model.query.MessageTemplateQuery;
import com.pei.dehaze.model.vo.MessageTemplateDetailVO;
import com.pei.dehaze.model.vo.MessageTemplateVO;

public interface MessageTemplateService extends IService<SysMessageTemplate> {

    Page<MessageTemplateVO> getPage(MessageTemplateQuery query);

    MessageTemplateDetailVO getDetail(Long id);

    void update(Long id, MessageTemplateForm form);

    SysMessageTemplate getByCode(String code);
}
