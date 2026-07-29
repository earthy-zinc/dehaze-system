package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.entity.SysMessage;
import com.pei.dehaze.model.form.MessageSendForm;
import com.pei.dehaze.model.query.MessageQuery;
import com.pei.dehaze.model.query.MessageSearchQuery;
import com.pei.dehaze.model.vo.*;

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
