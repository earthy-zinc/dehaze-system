package com.pei.clarity.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.clarity.model.entity.SysRating;
import com.pei.clarity.model.form.RatingCreateForm;
import com.pei.clarity.model.query.RatingPageQuery;
import com.pei.clarity.model.vo.IdVO;
import com.pei.clarity.model.vo.MyRatingVO;
import com.pei.clarity.model.vo.RatingDetailVO;
import com.pei.clarity.model.vo.RatingPageVO;
import com.pei.clarity.model.vo.RatingStatsVO;
import java.time.LocalDateTime;

public interface RatingService extends IService<SysRating> {

    IdVO createRating(RatingCreateForm form);

    void updateRating(Long id, RatingCreateForm form);

    RatingDetailVO getRatingByPrediction(Long predLogId);

    Page<MyRatingVO> listMyRatings(int pageNum, int pageSize);

    Page<RatingPageVO> listPagedRatings(RatingPageQuery query);

    void hideRating(Long id);

    void replyRating(Long id, String content);

    RatingStatsVO getRatingStats(LocalDateTime startTime, LocalDateTime endTime);
}
