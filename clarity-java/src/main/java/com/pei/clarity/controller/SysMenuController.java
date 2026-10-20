package com.pei.clarity.controller;

import com.pei.clarity.common.exception.BusinessException;
import com.pei.clarity.common.model.Option;
import com.pei.clarity.common.result.Result;
import com.pei.clarity.common.result.ResultCode;
import com.pei.clarity.model.form.MenuForm;
import com.pei.clarity.model.form.MenuVisibleBody;
import com.pei.clarity.model.query.MenuQuery;
import com.pei.clarity.model.vo.MenuVO;
import com.pei.clarity.model.vo.RouteVO;
import com.pei.clarity.plugin.dupsubmit.annotation.PreventDuplicateSubmit;
import com.pei.clarity.service.SysMenuService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import java.util.Arrays;
import java.util.List;
import lombok.RequiredArgsConstructor;
import org.springdoc.core.annotations.ParameterObject;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

/**
 * 菜单控制器
 *
 * @author earthyzinc
 * @since 2020/11/06
 */
@Tag(name = "04.菜单接口")
@RestController
@RequestMapping("/api/v1/menus")
@RequiredArgsConstructor
@Validated
public class SysMenuController {

    private final SysMenuService menuService;

    @Operation(summary = "菜单列表")
    @GetMapping
    public Result<List<MenuVO>> listMenus(@Valid @ParameterObject MenuQuery queryParams) {
        List<MenuVO> menuList = menuService.listMenus(queryParams);
        return Result.success(menuList);
    }

    @Operation(summary = "菜单下拉列表")
    @GetMapping("/options")
    public Result<List<Option<Long>>> listMenuOptions() {
        List<Option<Long>> menus = menuService.listMenuOptions();
        return Result.success(menus);
    }

    @Operation(summary = "路由列表")
    @GetMapping("/routes")
    public Result<List<RouteVO>> listRoutes() {
        List<RouteVO> routeList = menuService.listRoutes();
        return Result.success(routeList);
    }

    @Operation(summary = "菜单表单数据")
    @GetMapping("/{id}/form")
    public Result<MenuForm> getMenuForm(@Parameter(description = "菜单ID") @PathVariable Long id) {
        MenuForm menu = menuService.getMenuForm(id);
        return Result.success(menu);
    }

    @Operation(summary = "新增菜单")
    @PostMapping
    @PreAuthorize("@ss.hasPerm('sys:menu:add')")
    @PreventDuplicateSubmit
    public Result<Void> addMenu(@Valid @RequestBody MenuForm menuForm) {
        boolean result = menuService.saveMenu(menuForm);
        return Result.judge(result);
    }

    @Operation(summary = "修改菜单")
    @PutMapping(value = "/{id}")
    @PreAuthorize("@ss.hasPerm('sys:menu:edit')")
    public Result<Void> updateMenu(
            @Parameter(description = "菜单ID") @PathVariable Long id, @Valid @RequestBody MenuForm menuForm) {
        menuForm.setId(id);
        boolean result = menuService.saveMenu(menuForm);
        return Result.judge(result);
    }

    @Operation(summary = "删除菜单")
    @DeleteMapping("/{ids}")
    @PreAuthorize("@ss.hasPerm('sys:menu:delete')")
    public Result<Void> deleteMenu(@Parameter(description = "菜单ID，多个以英文(,)分割") @PathVariable("ids") String ids) {
        List<Long> idList;
        try {
            idList = Arrays.stream(ids.split(","))
                    .map(String::trim)
                    .map(Long::valueOf)
                    .toList();
        } catch (NumberFormatException e) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "菜单ID格式错误");
        }
        menuService.deleteMenu(idList);
        return Result.success();
    }

    @Operation(summary = "修改菜单显示状态")
    @PatchMapping("/{menuId}")
    @PreAuthorize("@ss.hasPerm('sys:menu:edit')")
    public Result<Void> updateMenuVisible(
            @Parameter(description = "菜单ID") @PathVariable Long menuId, @Valid @RequestBody MenuVisibleBody body) {
        boolean result = menuService.updateMenuVisible(menuId, body.getVisible());
        return Result.judge(result);
    }
}
