package com.pei.dehaze.controller;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.pei.dehaze.common.result.PageResult;
import com.pei.dehaze.model.query.AiCompatCallQuery;
import com.pei.dehaze.model.vo.AiCompatCallVO;
import com.pei.dehaze.service.AiCompatCallService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * AI 兼容 API 调用审计（登录用户仅可查本人），支撑兼容端点调用对账与异常排查。
 *
 * @author dehaze
 */
@Tag(name = "32.AI兼容调用审计")
@RestController
@RequestMapping("/api/v1/ai/compat")
@RequiredArgsConstructor
public class AiCompatCallController {

    private final AiCompatCallService compatCallService;

    @Operation(summary = "兼容调用审计查询（分页，当前用户）")
    @GetMapping("/calls")
    public PageResult<AiCompatCallVO> listCalls(@Valid @ParameterObject AiCompatCallQuery query) {
        IPage<AiCompatCallVO> page = compatCallService.listCalls(query);
        return PageResult.success(page);
    }
}
