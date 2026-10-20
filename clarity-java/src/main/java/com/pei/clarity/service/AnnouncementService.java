package com.pei.clarity.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.clarity.model.entity.SysAnnouncement;
import com.pei.clarity.model.form.AnnouncementForm;
import com.pei.clarity.model.query.AnnouncementQuery;
import com.pei.clarity.model.vo.AnnouncementDetailVO;
import com.pei.clarity.model.vo.AnnouncementSendResultVO;
import com.pei.clarity.model.vo.AnnouncementVO;

public interface AnnouncementService extends IService<SysAnnouncement> {

    Long create(AnnouncementForm form);

    Page<AnnouncementVO> getPage(AnnouncementQuery query);

    AnnouncementDetailVO getDetail(Long id);

    void update(Long id, AnnouncementForm form);

    void delete(Long id);

    AnnouncementSendResultVO send(Long id);

    void cancel(Long id);
}
