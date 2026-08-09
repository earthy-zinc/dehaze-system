package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.entity.SysInputHistory;
import com.pei.dehaze.model.form.HistoryForm;
import com.pei.dehaze.model.query.HistoryQuery;
import com.pei.dehaze.model.vo.InputHistoryVO;

import java.util.List;

/**
 * 图像输入历史记录服务
 *
 * @author earthyzinc
 * @since 2024-06-12
 */
public interface SysInputHistoryService extends IService<SysInputHistory> {

    /** 分页查询当前用户的历史记录 */
    Page<InputHistoryVO> getHistoryPage(HistoryQuery query);

    /** 获取历史记录详情 */
    InputHistoryVO getHistoryById(Long id);

    /** 创建历史记录 */
    Long createHistory(HistoryForm form);

    /** 删除单条历史记录 */
    boolean deleteHistory(Long id);

    /** 批量删除历史记录 */
    int batchDeleteHistory(List<Long> ids);

    /** 清空当前用户的所有历史记录 */
    int clearAllHistory();
}
