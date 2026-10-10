package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.entity.SysMember;
import com.pei.dehaze.model.entity.SysMemberBenefit;
import com.pei.dehaze.model.form.BenefitForm;
import com.pei.dehaze.model.vo.BenefitVO;
import java.util.List;

public interface MemberBenefitService extends IService<SysMemberBenefit> {

    SysMemberBenefit getByLevelCode(String levelCode);

    List<SysMemberBenefit> listAllOrdered();

    List<BenefitVO> listVOs();

    void updateByLevelCode(String levelCode, BenefitForm form);

    /** 按会员当前等级权益 + 已购会员卡覆盖项回刷 8 类配额快照字段（只写实体，不落库、不删缓存） */
    void refreshMemberQuotaSnapshot(SysMember member);

    /** 同上，但写入 UpdateWrapper（订单履约按用户部分更新，不回写整行） */
    void appendQuotaSnapshot(LambdaUpdateWrapper<SysMember> wrapper, SysMember member, SysMemberBenefit benefit);

    /** 失效会员配额缓存：member:level:{userId} + 8 类 member:quota:{userId}:{taskType} */
    void invalidateMemberQuotaCache(Long userId);
}
