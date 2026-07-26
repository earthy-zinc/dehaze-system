package com.pei.dehaze.service.importexport.handler;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.pei.dehaze.common.base.IBaseEnum;
import com.pei.dehaze.common.enums.StatusEnum;
import com.pei.dehaze.model.query.UserPageQuery;
import com.pei.dehaze.model.vo.UserPageVO;
import com.pei.dehaze.service.SysUserService;
import com.pei.dehaze.service.importexport.ExportHandler;
import com.pei.dehaze.service.importexport.model.ExportContext;
import com.pei.dehaze.service.importexport.model.ExportDataProvider;
import com.pei.dehaze.service.importexport.model.ExportFieldConfig;
import com.pei.dehaze.service.strategy.ProgressCallback;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 用户导出处理器
 * <p>复用 {@link SysUserService#listPagedUsers} 分页查询，流式输出到 Excel/CSV。
 */
@Component
@RequiredArgsConstructor
public class UserExportHandler implements ExportHandler {

    private static final int PAGE_SIZE = 1000;

    private final SysUserService userService;

    @Override
    public String getModule() {
        return "user";
    }

    @Override
    public long estimateCount(Map<String, Object> queryParams) {
        UserPageQuery query = buildQuery(queryParams, 1, 1);
        return userService.listPagedUsers(query).getTotal();
    }

    @Override
    public void export(ExportContext ctx, ProgressCallback callback) throws Exception {
        // 通用流程通过 getDataProvider + fileGenerator 写入，此处无需实现
    }

    @Override
    public List<ExportFieldConfig> getFieldConfigs() {
        return List.of(
                ExportFieldConfig.builder().field("username").label("用户名").order(1).build(),
                ExportFieldConfig.builder().field("nickname").label("昵称").order(2).build(),
                ExportFieldConfig.builder().field("deptName").label("部门").order(3).build(),
                ExportFieldConfig.builder().field("genderLabel").label("性别").order(4).build(),
                ExportFieldConfig.builder().field("mobile").label("手机号").order(5).build(),
                ExportFieldConfig.builder().field("email").label("邮箱").order(6).build(),
                ExportFieldConfig.builder().field("statusLabel").label("状态").order(7).build(),
                ExportFieldConfig.builder().field("createTime").label("创建时间").order(8)
                        .dateFormat("yyyy-MM-dd HH:mm:ss").build()
        );
    }

    @Override
    public ExportDataProvider getDataProvider(ExportContext ctx) {
        List<ExportFieldConfig> fields = ctx.getSelectedFields() == null || ctx.getSelectedFields().isEmpty()
                ? getFieldConfigs()
                : getFieldConfigs().stream()
                .filter(f -> ctx.getSelectedFields().contains(f.getField()))
                .toList();

        return (pageNum, pageSize) -> {
            UserPageQuery query = buildQuery(ctx.getQueryParams(), pageNum, PAGE_SIZE);
            IPage<UserPageVO> page = userService.listPagedUsers(query);
            if (page.getRecords() == null || page.getRecords().isEmpty()) {
                return List.of();
            }
            List<List<Object>> rows = new ArrayList<>(page.getRecords().size());
            for (UserPageVO vo : page.getRecords()) {
                rows.add(toRow(vo, fields));
            }
            return rows;
        };
    }

    private UserPageQuery buildQuery(Map<String, Object> params, int pageNum, int pageSize) {
        UserPageQuery query = new UserPageQuery();
        query.setPageNum(pageNum);
        query.setPageSize(pageSize);
        if (params == null) {
            return query;
        }
        Object keywords = params.get("keywords");
        if (keywords != null) {
            query.setKeywords(String.valueOf(keywords));
        }
        Object status = params.get("status");
        if (status != null && !"".equals(String.valueOf(status))) {
            query.setStatus(Integer.valueOf(String.valueOf(status)));
        }
        Object deptId = params.get("deptId");
        if (deptId != null && !"".equals(String.valueOf(deptId))) {
            query.setDeptId(Long.valueOf(String.valueOf(deptId)));
        }
        Object startTime = params.get("startTime");
        if (startTime != null) {
            query.setStartTime(String.valueOf(startTime));
        }
        Object endTime = params.get("endTime");
        if (endTime != null) {
            query.setEndTime(String.valueOf(endTime));
        }
        return query;
    }

    private List<Object> toRow(UserPageVO vo, List<ExportFieldConfig> fields) {
        List<Object> row = new ArrayList<>(fields.size());
        for (ExportFieldConfig f : fields) {
            row.add(extractField(vo, f));
        }
        return row;
    }

    private Object extractField(UserPageVO vo, ExportFieldConfig f) {
        return switch (f.getField()) {
            case "username" -> nullToEmpty(vo.getUsername());
            case "nickname" -> nullToEmpty(vo.getNickname());
            case "deptName" -> nullToEmpty(vo.getDeptName());
            case "genderLabel" -> nullToEmpty(vo.getGenderLabel());
            case "mobile" -> nullToEmpty(vo.getMobile());
            case "email" -> nullToEmpty(vo.getEmail());
            case "statusLabel" -> vo.getStatus() == null
                    ? ""
                    : IBaseEnum.getLabelByValue(vo.getStatus(), StatusEnum.class);
            case "createTime" -> vo.getCreateTime() == null
                    ? ""
                    : new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(vo.getCreateTime());
            default -> "";
        };
    }

    private String nullToEmpty(String s) {
        return s == null ? "" : s;
    }
}
