package com.pei.dehaze.service.impl;

import cn.hutool.core.text.CharSequenceUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.fasterxml.jackson.core.type.TypeReference;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.common.util.AiJsonUtils;
import com.pei.dehaze.mapper.SysAiSkillFileMapper;
import com.pei.dehaze.mapper.SysAiSkillMapper;
import com.pei.dehaze.model.entity.SysAiSkill;
import com.pei.dehaze.model.entity.SysAiSkillFile;
import com.pei.dehaze.model.form.SkillForm;
import com.pei.dehaze.model.form.SkillUpdateForm;
import com.pei.dehaze.model.query.SkillPageQuery;
import com.pei.dehaze.model.vo.SkillListItemVO;
import com.pei.dehaze.model.vo.SkillMarketVO;
import com.pei.dehaze.model.vo.SkillVO;
import com.pei.dehaze.security.util.SecurityUtils;
import com.pei.dehaze.service.AiSkillService;
import com.pei.dehaze.service.impl.file.StorageServiceFactory;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

@Slf4j
@Service
@RequiredArgsConstructor
public class AiSkillServiceImpl extends ServiceImpl<SysAiSkillMapper, SysAiSkill> implements AiSkillService {

    private static final String MANAGE_PERMISSION = "ai:skill:manage";
    private static final int STATUS_ENABLED = 1;
    private static final int STATUS_DISABLED = 0;

    /** 指令内容上限 100KB */
    private static final int CONTENT_MAX_BYTES = 100 * 1024;

    /** 危险操作正则（命中即拦截，防止 Skill 指令被注入破坏性 shell 命令），与 python 同源 */
    private static final Pattern DANGEROUS_PATTERN = Pattern.compile(
            "(rm\\s+-rf\\s*/|mkfs\\.?(ext\\d?|xfs|vfat|ntfs)?\\b|curl[^\\n]*\\|\\s*(ba)?sh\\b"
                    + "|wget[^\\n]*\\|\\s*(ba)?sh\\b|sudo\\s+(rm|shutdown|reboot|mkfs|dd)|dd\\s+if=.*of=/dev/)",
            Pattern.CASE_INSENSITIVE);

    /** SKILL 目录文件对象 key 前缀：skills/{skill_id}/{path}（用 id 而非 name，改名不影响定位） */
    private static final String SKILL_OBJECT_PREFIX = "skills";

    private final SysAiSkillFileMapper skillFileMapper;
    private final StorageServiceFactory storageServiceFactory;

    @Override
    @Transactional(readOnly = true)
    public Page<SkillListItemVO> listSkills(SkillPageQuery query) {
        if (!isManager()) {
            // 普通用户仅返回启用项（不分页，直接全量）
            List<SysAiSkill> skills = this.list(new LambdaQueryWrapper<SysAiSkill>()
                    .eq(SysAiSkill::getStatus, STATUS_ENABLED)
                    .orderByAsc(SysAiSkill::getId));
            List<SkillListItemVO> items = toListItems(skills);
            Page<SkillListItemVO> page = new Page<>(1, items.size(), items.size());
            page.setRecords(items);
            return page;
        }
        LambdaQueryWrapper<SysAiSkill> wrapper = new LambdaQueryWrapper<SysAiSkill>()
                .like(CharSequenceUtil.isNotBlank(query.getKeyword()), SysAiSkill::getName, query.getKeyword())
                .eq(query.getStatus() != null, SysAiSkill::getStatus, query.getStatus())
                .orderByDesc(SysAiSkill::getId);
        Page<SysAiSkill> page = this.page(new Page<>(query.getPageNum(), query.getPageSize()), wrapper);
        Page<SkillListItemVO> result = new Page<>(page.getCurrent(), page.getSize(), page.getTotal());
        result.setRecords(toListItems(page.getRecords()));
        return result;
    }

    @Override
    @Transactional(readOnly = true)
    public List<SkillMarketVO> listMarket() {
        List<SysAiSkill> skills = this.list(new LambdaQueryWrapper<SysAiSkill>()
                .eq(SysAiSkill::getMarketShared, 1)
                .eq(SysAiSkill::getStatus, STATUS_ENABLED)
                .orderByAsc(SysAiSkill::getId));
        Map<String, Integer> refs = countAgentReferences(skills);
        List<SkillMarketVO> items = new ArrayList<>(skills.size());
        for (SysAiSkill skill : skills) {
            SkillMarketVO vo = new SkillMarketVO();
            vo.setSkillId(skill.getId());
            vo.setName(skill.getName());
            vo.setDescription(skill.getDescription());
            vo.setScene(skill.getScene() == null ? "" : skill.getScene());
            vo.setEnabled(skill.getStatus() != null && skill.getStatus() == STATUS_ENABLED);
            vo.setAgentCount(refs.getOrDefault(skill.getName(), 0));
            items.add(vo);
        }
        return items;
    }

    @Override
    @Transactional(readOnly = true)
    public SkillVO getSkill(Long skillId) {
        return toDetail(getOrRaise(skillId), null);
    }

    @Override
    @Transactional(readOnly = true)
    public byte[] getSkillFile(Long skillId, String path) {
        SysAiSkill skill = getOrRaise(skillId);
        List<SysAiSkillFile> files = listFiles(skill.getId());
        boolean exists = files.stream().anyMatch(file -> file.getPath().equals(path));
        if (!exists) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "SKILL 文件不存在");
        }
        try (InputStream stream = storageServiceFactory.getDefault().downLoadFile(objectKey(skill.getId(), path));
             ByteArrayOutputStream buffer = new ByteArrayOutputStream()) {
            stream.transferTo(buffer);
            return buffer.toByteArray();
        } catch (Exception e) {
            log.warn("SKILL 文件读取失败 object={}: {}", objectKey(skill.getId(), path), e.getMessage());
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "SKILL 文件读取失败");
        }
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public SkillVO createSkill(SkillForm form) {
        validateContent(form.getInstruction());
        if (getByName(form.getName()) != null) {
            throw new BusinessException(ResultCode.DATA_EXISTS, "Skill 名称已存在");
        }
        SysAiSkill skill = new SysAiSkill();
        skill.setName(form.getName());
        skill.setDescription(form.getDescription());
        skill.setScene(form.getScene());
        skill.setInstruction(form.getInstruction());
        // 创建后为禁用态，需管理员显式启用才可被加载（§2.6.11"启用才可被使用"）
        skill.setStatus(STATUS_DISABLED);
        skill.setSource("admin");
        skill.setMarketShared(0);
        this.save(skill);
        return toDetail(skill, null);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public SkillVO updateSkill(Long skillId, SkillUpdateForm form) {
        SysAiSkill skill = getOrRaise(skillId);
        if (form.getName() != null && !form.getName().equals(skill.getName())) {
            SysAiSkill duplicate = getByName(form.getName());
            if (duplicate != null && !duplicate.getId().equals(skillId)) {
                throw new BusinessException(ResultCode.DATA_EXISTS, "Skill 名称已存在");
            }
            skill.setName(form.getName());
        }
        if (form.getInstruction() != null) {
            validateContent(form.getInstruction());
            skill.setInstruction(form.getInstruction());
        }
        if (form.getDescription() != null) {
            skill.setDescription(form.getDescription());
        }
        if (form.getScene() != null) {
            skill.setScene(form.getScene());
        }
        this.updateById(skill);
        return toDetail(skill, null);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public SkillVO setStatus(Long skillId, boolean enabled) {
        SysAiSkill skill = getOrRaise(skillId);
        int target = enabled ? STATUS_ENABLED : STATUS_DISABLED;
        if (skill.getStatus() == null || skill.getStatus() != target) {
            skill.setStatus(target);
            this.updateById(skill);
        }
        return toDetail(skill, null);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void deleteSkill(Long skillId) {
        SysAiSkill skill = getOrRaise(skillId);
        long refs = this.baseMapper.countAgentReferences(skill.getName());
        if (refs > 0) {
            throw new BusinessException(ResultCode.DATA_BIND_EXISTS,
                    "Skill [" + skill.getName() + "] 已被 " + refs + " 个 Agent 关联，请先解绑再删除");
        }
        this.removeById(skillId);
        // 对象存储资源与文件清单清理（软删主表后清理，失败不阻断删除语义）
        for (SysAiSkillFile file : listFiles(skillId)) {
            try {
                storageServiceFactory.getDefault().deleteFile(objectKey(skillId, file.getPath()));
            } catch (Exception e) {
                log.warn("SKILL 对象存储清理失败 object={}: {}", objectKey(skillId, file.getPath()), e.getMessage());
            }
            skillFileMapper.deleteById(file.getId());
        }
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public SkillVO shareToMarket(Long skillId) {
        SysAiSkill skill = getOrRaise(skillId);
        if (skill.getStatus() == null || skill.getStatus() != STATUS_ENABLED) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "Skill 需先启用才能共享至市场");
        }
        if (skill.getMarketShared() == null || skill.getMarketShared() != 1) {
            skill.setMarketShared(1);
            this.updateById(skill);
        }
        return toDetail(skill, null);
    }

    // ==================== 内部实现 ====================

    /** 管理员判定：ROOT 或持有 ai:skill:manage（与 python _is_manager 同口径） */
    private boolean isManager() {
        return SecurityUtils.isRoot() || SecurityUtils.getPerms().contains(MANAGE_PERMISSION);
    }

    private SysAiSkill getOrRaise(Long skillId) {
        SysAiSkill skill = this.getById(skillId);
        boolean visible = skill != null && (isManager()
                || (skill.getStatus() != null && skill.getStatus() == STATUS_ENABLED));
        if (!visible) {
            // 禁用项对普通用户按不存在处理，防止经 ID 直读禁用 Skill 的指令全文
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "Skill 不存在");
        }
        return skill;
    }

    private SysAiSkill getByName(String name) {
        return this.getOne(new LambdaQueryWrapper<SysAiSkill>()
                .eq(SysAiSkill::getName, name)
                .orderByAsc(SysAiSkill::getId), false);
    }

    /** 指令内容校验：长度上限（100KB）+ 危险操作拦截 */
    private void validateContent(String content) {
        if (content.getBytes(StandardCharsets.UTF_8).length > CONTENT_MAX_BYTES) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "Skill 指令内容超过 100KB 上限");
        }
        if (DANGEROUS_PATTERN.matcher(content).find()) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "指令含危险操作");
        }
    }

    private List<SysAiSkillFile> listFiles(Long skillId) {
        return skillFileMapper.selectList(new LambdaQueryWrapper<SysAiSkillFile>()
                .eq(SysAiSkillFile::getSkillId, skillId)
                .orderByAsc(SysAiSkillFile::getPath));
    }

    private Map<String, Integer> countAgentReferences(List<SysAiSkill> skills) {
        Map<String, Integer> refs = new HashMap<>();
        if (skills.isEmpty()) {
            return refs;
        }
        List<String> names = skills.stream().map(SysAiSkill::getName).toList();
        for (Map<String, Object> row : this.baseMapper.countAgentReferencesByNames(names)) {
            Object name = row.get("skillName");
            Object total = row.get("total");
            if (name != null) {
                refs.put(String.valueOf(name), total instanceof Number number ? number.intValue() : 0);
            }
        }
        return refs;
    }

    private List<SkillListItemVO> toListItems(List<SysAiSkill> skills) {
        Map<String, Integer> refs = countAgentReferences(skills);
        List<SkillListItemVO> items = new ArrayList<>(skills.size());
        for (SysAiSkill skill : skills) {
            SkillListItemVO vo = new SkillListItemVO();
            vo.setId(skill.getId());
            vo.setName(skill.getName());
            vo.setDescription(skill.getDescription());
            vo.setScene(skill.getScene());
            vo.setStatus(skill.getStatus());
            vo.setSource(skill.getSource());
            vo.setMarketShared(skill.getMarketShared());
            vo.setAgentCount(refs.getOrDefault(skill.getName(), 0));
            vo.setCreateTime(skill.getCreateTime());
            vo.setUpdateTime(skill.getUpdateTime());
            items.add(vo);
        }
        return items;
    }

    private SkillVO toDetail(SysAiSkill skill, List<SkillVO.SkippedFile> skipped) {
        SkillVO vo = new SkillVO();
        vo.setId(skill.getId());
        vo.setName(skill.getName());
        vo.setDescription(skill.getDescription());
        vo.setScene(skill.getScene());
        vo.setInstruction(skill.getInstruction());
        vo.setLicense(skill.getLicense());
        vo.setCompatibility(skill.getCompatibility());
        vo.setMetadata(AiJsonUtils.read(skill.getMetadata(), new TypeReference<Map<String, Object>>() {
        }));
        vo.setAllowedTools(skill.getAllowedTools());
        vo.setStatus(skill.getStatus());
        vo.setSource(skill.getSource());
        vo.setMarketShared(skill.getMarketShared());
        vo.setCreateTime(skill.getCreateTime());
        vo.setUpdateTime(skill.getUpdateTime());
        vo.setAgentCount((int) this.baseMapper.countAgentReferences(skill.getName()));

        List<SkillVO.FileItem> files = new ArrayList<>();
        for (SysAiSkillFile file : listFiles(skill.getId())) {
            SkillVO.FileItem item = new SkillVO.FileItem();
            item.setPath(file.getPath());
            item.setFileSize(file.getFileSize());
            item.setFileType(file.getFileType());
            files.add(item);
        }
        vo.setFiles(files);
        vo.setSkippedFiles(skipped == null ? new ArrayList<>() : skipped);
        return vo;
    }

    private static String objectKey(Long skillId, String relPath) {
        return SKILL_OBJECT_PREFIX + "/" + skillId + "/" + relPath;
    }
}
