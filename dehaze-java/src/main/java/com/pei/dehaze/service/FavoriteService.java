package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.entity.SysFavorite;
import com.pei.dehaze.model.form.FavoriteForm;
import com.pei.dehaze.model.query.FavoritePageQuery;
import com.pei.dehaze.model.vo.FavoriteCountVO;
import com.pei.dehaze.model.vo.FavoriteStatusVO;
import com.pei.dehaze.model.vo.FavoriteVO;

import java.util.List;

public interface FavoriteService extends IService<SysFavorite> {

    /**
     * 添加收藏，返回收藏记录ID
     */
    Long add(FavoriteForm form);

    /**
     * 批量取消收藏（逻辑删除）
     */
    void deleteByIds(List<Long> ids);

    /**
     * 收藏列表分页查询
     */
    Page<FavoriteVO> getPage(FavoritePageQuery query);

    /**
     * 检查指定对象是否已收藏
     */
    FavoriteStatusVO getStatus(String targetType, Long targetId);

    /**
     * 收藏数量统计（按类型分组）
     */
    List<FavoriteCountVO> getCount(String targetType);

    /**
     * 标记指定对象的收藏记录为已失效（供算法/数据集等模块删除时回调）
     */
    void markInvalid(String targetType, List<Long> targetIds);
}
