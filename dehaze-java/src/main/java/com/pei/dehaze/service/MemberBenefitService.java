package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.entity.SysMemberBenefit;
import com.pei.dehaze.model.form.BenefitForm;
import com.pei.dehaze.model.vo.BenefitVO;

import java.util.List;

public interface MemberBenefitService extends IService<SysMemberBenefit> {

    SysMemberBenefit getByLevelCode(String levelCode);

    List<SysMemberBenefit> listAllOrdered();

    List<BenefitVO> listVOs();

    void updateByLevelCode(String levelCode, BenefitForm form);
}
