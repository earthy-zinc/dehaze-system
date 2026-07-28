package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.entity.SysFeedback;
import com.pei.dehaze.model.form.FeedbackAssignForm;
import com.pei.dehaze.model.form.FeedbackCloseForm;
import com.pei.dehaze.model.form.FeedbackCreateForm;
import com.pei.dehaze.model.form.FeedbackReplyForm;
import com.pei.dehaze.model.form.FeedbackSupplementForm;
import com.pei.dehaze.model.query.FeedbackPageQuery;
import com.pei.dehaze.model.vo.FeedbackDetailVO;
import com.pei.dehaze.model.vo.FeedbackPageVO;
import com.pei.dehaze.model.vo.FeedbackStatsVO;
import com.pei.dehaze.model.vo.IdVO;

import java.time.LocalDateTime;
import java.util.List;

public interface FeedbackService extends IService<SysFeedback> {

    IdVO createFeedback(FeedbackCreateForm form);

    Page<FeedbackPageVO> listMyFeedback(int pageNum, int pageSize);

    FeedbackDetailVO getFeedbackDetail(Long id);

    void supplementFeedback(Long id, FeedbackSupplementForm form);

    Page<FeedbackPageVO> listPagedFeedback(FeedbackPageQuery query);

    void assignFeedback(Long id, FeedbackAssignForm form);

    void replyFeedback(Long id, FeedbackReplyForm form);

    void closeFeedback(Long id, FeedbackCloseForm form);

    void updateTags(Long id, List<String> tags);

    FeedbackStatsVO getFeedbackStats(LocalDateTime startTime, LocalDateTime endTime);
}
