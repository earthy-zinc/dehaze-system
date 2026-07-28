package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.entity.SysRating;
import com.pei.dehaze.model.form.RatingCreateForm;
import com.pei.dehaze.model.query.RatingPageQuery;
import com.pei.dehaze.model.vo.IdVO;
import com.pei.dehaze.model.vo.MyRatingVO;
import com.pei.dehaze.model.vo.RatingDetailVO;
import com.pei.dehaze.model.vo.RatingPageVO;
import com.pei.dehaze.model.vo.RatingStatsVO;

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
