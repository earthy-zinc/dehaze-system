package com.pei.dehaze.ui.system.viewmodel;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.sdk.api.AnnouncementAPI;
import com.pei.dehaze.sdk.model.message.AnnouncementQuery;
import com.pei.dehaze.sdk.model.message.AnnouncementVO;

/**
 * 消息管理 ViewModel（公告管理）
 */
public class MessageManageViewModel extends BaseManageViewModel<AnnouncementVO> {

    @Override
    public void loadData() {
        AnnouncementQuery query = new AnnouncementQuery();
        query.setPageNum(pageNum);
        query.setPageSize(pageSize);
        query.setTitle(keywords.isEmpty() ? null : keywords);
        AnnouncementAPI.getPage(query, RepositoryAdapters.wrap(withLoading(data -> {
            itemList.postValue(data.getList());
            total.postValue(data.getTotal());
        })));
    }
}
