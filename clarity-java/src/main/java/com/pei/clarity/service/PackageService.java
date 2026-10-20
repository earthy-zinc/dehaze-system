package com.pei.clarity.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.clarity.model.entity.SysPackage;
import com.pei.clarity.model.form.PackageForm;
import com.pei.clarity.model.query.PackagePageQuery;
import com.pei.clarity.model.vo.PackageDetailVO;
import com.pei.clarity.model.vo.PackagePageVO;
import com.pei.clarity.model.vo.PriceResult;
import com.pei.clarity.model.vo.SalesStatsVO;
import java.util.List;

public interface PackageService extends IService<SysPackage> {

    List<PackageDetailVO> listOnSale();

    PackageDetailVO getDetail(Long id);

    Page<PackagePageVO> getPage(PackagePageQuery query);

    PackageForm getForm(Long id);

    void save(PackageForm form);

    void update(Long id, PackageForm form);

    void deleteByIds(String ids);

    void updateStatus(Long id, Integer status);

    PriceResult calculatePrice(Long packageId, Long userCouponId);

    SalesStatsVO getSalesStats();
}
