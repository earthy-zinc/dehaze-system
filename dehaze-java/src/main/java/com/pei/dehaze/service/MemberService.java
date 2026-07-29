package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.entity.SysMember;
import com.pei.dehaze.model.form.MemberGrowthAdjustForm;
import com.pei.dehaze.model.form.MemberLevelAdjustForm;
import com.pei.dehaze.model.form.MemberStatusForm;
import com.pei.dehaze.model.query.GrowthLogQuery;
import com.pei.dehaze.model.query.MemberPageQuery;
import com.pei.dehaze.model.vo.*;

public interface MemberService extends IService<SysMember> {

    MemberProfileVO getProfile();

    Page<MemberPageVO> getPage(MemberPageQuery query);

    MemberDetailVO getDetail(Long userId);

    void adjustLevel(Long userId, MemberLevelAdjustForm form);

    void adjustGrowth(Long userId, MemberGrowthAdjustForm form);

    void updateStatus(Long userId, MemberStatusForm form);

    Page<GrowthLogVO> getGrowthLogs(GrowthLogQuery query);

    SignInResultVO signIn();

    SignInCalendarVO getSignInCalendar(Integer year, Integer month);

    void resetMonthlyQuota();

    void processExpiredMembers();

    boolean deductQuota(Long userId, String quotaType, int amount);

    void initMember(Long userId);

    void sendExpireReminders();
}
