package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.entity.SysAiSkill;
import com.pei.dehaze.model.form.SkillForm;
import com.pei.dehaze.model.form.SkillUpdateForm;
import com.pei.dehaze.model.query.SkillPageQuery;
import com.pei.dehaze.model.vo.SkillListItemVO;
import com.pei.dehaze.model.vo.SkillMarketVO;
import com.pei.dehaze.model.vo.SkillVO;

import java.util.List;

/** Skills 管理（F-M08-006 §2.6.11/§2.6.14） */
public interface AiSkillService extends IService<SysAiSkill> {

    Page<SkillListItemVO> listSkills(SkillPageQuery query);

    List<SkillMarketVO> listMarket();

    SkillVO getSkill(Long skillId);

    /** 读取 SKILL 目录内资源文件内容（须命中该 Skill 的文件清单） */
    byte[] getSkillFile(Long skillId, String path);

    SkillVO createSkill(SkillForm form);

    SkillVO updateSkill(Long skillId, SkillUpdateForm form);

    SkillVO setStatus(Long skillId, boolean enabled);

    void deleteSkill(Long skillId);

    SkillVO shareToMarket(Long skillId);
}
