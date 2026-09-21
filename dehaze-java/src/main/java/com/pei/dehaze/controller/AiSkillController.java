package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.common.result.Result;
import com.pei.dehaze.model.form.SkillForm;
import com.pei.dehaze.model.form.SkillShareForm;
import com.pei.dehaze.model.form.SkillStatusForm;
import com.pei.dehaze.model.form.SkillUpdateForm;
import com.pei.dehaze.model.query.SkillPageQuery;
import com.pei.dehaze.model.vo.SkillListItemVO;
import com.pei.dehaze.model.vo.SkillMarketVO;
import com.pei.dehaze.model.vo.SkillVO;
import com.pei.dehaze.service.AiSkillService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.http.MediaType;
import org.springframework.http.MediaTypeFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@Tag(name = "Skills 管理")
@RestController
@RequestMapping("/api/v1/ai/skills")
@RequiredArgsConstructor
public class AiSkillController {

    private final AiSkillService aiSkillService;

    @Operation(summary = "Skills 列表")
    @GetMapping
    public PageResult<SkillListItemVO> listSkills(@Valid @ParameterObject SkillPageQuery query) {
        Page<SkillListItemVO> page = aiSkillService.listSkills(query);
        return PageResult.success(page);
    }

    @Operation(summary = "Skill 市场目录")
    @GetMapping("/market")
    public Result<List<SkillMarketVO>> listMarket() {
        return Result.success(aiSkillService.listMarket());
    }

    @Operation(summary = "Skill 详情")
    @GetMapping("/{skillId}")
    public Result<SkillVO> getSkill(@PathVariable Long skillId) {
        return Result.success(aiSkillService.getSkill(skillId));
    }

    @Operation(summary = "读取 SKILL 资源文件")
    @GetMapping("/{skillId}/file")
    public ResponseEntity<byte[]> getSkillFile(@PathVariable Long skillId,
                                               @RequestParam String path) {
        byte[] content = aiSkillService.getSkillFile(skillId, path);
        MediaType mediaType = MediaTypeFactory.getMediaType(path)
                .orElse(MediaType.APPLICATION_OCTET_STREAM);
        return ResponseEntity.ok()
                .contentType(mediaType)
                .body(content);
    }

    @Operation(summary = "创建 Skill")
    @PostMapping
    @PreAuthorize("@ss.hasPerm('ai:skill:manage')")
    public Result<SkillVO> createSkill(@RequestBody @Valid SkillForm form) {
        return Result.success(aiSkillService.createSkill(form));
    }

    @Operation(summary = "更新 Skill")
    @PutMapping("/{skillId}")
    @PreAuthorize("@ss.hasPerm('ai:skill:manage')")
    public Result<SkillVO> updateSkill(@PathVariable Long skillId,
                                       @RequestBody @Valid SkillUpdateForm form) {
        return Result.success(aiSkillService.updateSkill(skillId, form));
    }

    @Operation(summary = "共享 Skill 至市场")
    @PostMapping("/market")
    @PreAuthorize("@ss.hasPerm('ai:skill:manage')")
    public Result<SkillVO> shareToMarket(@RequestBody @Valid SkillShareForm form) {
        return Result.success(aiSkillService.shareToMarket(form.getSkillId()));
    }

    @Operation(summary = "启停 Skill")
    @PatchMapping("/{skillId}/status")
    @PreAuthorize("@ss.hasPerm('ai:skill:manage')")
    public Result<SkillVO> setSkillStatus(@PathVariable Long skillId,
                                          @RequestBody @Valid SkillStatusForm form) {
        return Result.success(aiSkillService.setStatus(skillId, form.getStatus() == 1));
    }

    @Operation(summary = "删除 Skill")
    @DeleteMapping("/{skillId}")
    @PreAuthorize("@ss.hasPerm('ai:skill:manage')")
    public Result<Void> deleteSkill(@PathVariable Long skillId) {
        aiSkillService.deleteSkill(skillId);
        return Result.success();
    }
}
