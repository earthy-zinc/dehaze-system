package com.pei.clarity.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.clarity.model.entity.SysMessage;
import com.pei.clarity.model.form.MessageSendForm;
import com.pei.clarity.model.query.MessageQuery;
import com.pei.clarity.model.query.MessageSearchQuery;
import com.pei.clarity.model.vo.*;

public interface MessageService extends IService<SysMessage> {

    MessageSendResultVO send(MessageSendForm form);

    Page<MessageVO> getPage(MessageQuery query);

    UnreadCountVO getUnreadCount();

    MessageDetailVO getDetail(Long id);

    void markRead(Long id);

    ReadAllResultVO markAllRead(String type);

    void deleteByIds(String ids);

    Page<MessageVO> search(MessageSearchQuery query);

    void refreshUnreadCountCache();
}
