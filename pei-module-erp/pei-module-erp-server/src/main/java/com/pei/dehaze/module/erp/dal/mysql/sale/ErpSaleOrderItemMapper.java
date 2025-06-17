package com.pei.dehaze.module.erp.dal.mysql.sale;

import com.pei.dehaze.framework.mybatis.core.mapper.BaseMapperX;
import com.pei.dehaze.module.erp.dal.dataobject.sale.ErpSaleOrderItemDO;
import org.apache.ibatis.annotations.Mapper;

import java.util.Collection;
import java.util.List;

/**
 * ERP 销售订单明项目 Mapper
 *
 * @author earthyzinc
 */
@Mapper
public interface ErpSaleOrderItemMapper extends BaseMapperX<ErpSaleOrderItemDO> {

    default List<ErpSaleOrderItemDO> selectListByOrderId(Long orderId) {
        return selectList(ErpSaleOrderItemDO::getOrderId, orderId);
    }

    default List<ErpSaleOrderItemDO> selectListByOrderIds(Collection<Long> orderIds) {
        return selectList(ErpSaleOrderItemDO::getOrderId, orderIds);
    }

    default int deleteByOrderId(Long orderId) {
        return delete(ErpSaleOrderItemDO::getOrderId, orderId);
    }

}
