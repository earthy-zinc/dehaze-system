package com.pei.dehaze.ui.system.viewmodel;

import com.pei.dehaze.repository.RepositoryAdapters;
import com.pei.dehaze.sdk.api.FeedbackAPI;
import com.pei.dehaze.sdk.model.feedback.FeedbackCloseForm;
import com.pei.dehaze.sdk.model.feedback.FeedbackPageVO;
import com.pei.dehaze.sdk.model.feedback.FeedbackQuery;
import com.pei.dehaze.sdk.model.feedback.FeedbackReplyForm;

/**
 * 反馈评价管理 ViewModel — 含列表查询、回复、关闭
 */
public class FeedbackManageViewModel extends BaseManageViewModel<FeedbackPageVO> {

    @Override
    public void loadData() {
        FeedbackQuery query = new FeedbackQuery();
        query.setPageNum(pageNum);
        query.setPageSize(pageSize);
        query.setKeywords(keywords.isEmpty() ? null : keywords);
        FeedbackAPI.listFeedback(query, RepositoryAdapters.wrap(withLoading(data -> {
            itemList.postValue(data.getList());
            total.postValue(data.getTotal());
        })));
    }

    public void replyFeedback(long id, String content) {
        FeedbackReplyForm form = new FeedbackReplyForm();
        form.setContent(content);
        form.setReplyType("text");
        FeedbackAPI.replyFeedback(id, form, RepositoryAdapters.wrap(withLoading(
                data -> {
                    operationResult.postValue("回复成功");
                    loadData();
                },
                errorMsg -> {
                    error.postValue(errorMsg);
                    loading.postValue(false);
                }
        )));
    }

    public void closeFeedback(long id, String reason) {
        FeedbackCloseForm form = new FeedbackCloseForm();
        form.setCloseReason(reason);
        FeedbackAPI.closeFeedback(id, form, RepositoryAdapters.wrap(withLoading(
                data -> {
                    operationResult.postValue("反馈已关闭");
                    loadData();
                },
                errorMsg -> {
                    error.postValue(errorMsg);
                    loading.postValue(false);
                }
        )));
    }
}
