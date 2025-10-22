package com.pei.dehaze.module.iot.service.product;

import cn.hutool.core.collection.CollUtil;
import com.pei.dehaze.framework.common.pojo.PageResult;
import com.pei.dehaze.framework.common.util.object.BeanUtils;
import com.pei.dehaze.module.iot.controller.admin.product.vo.category.IotProductCategoryPageReqVO;
import com.pei.dehaze.module.iot.controller.admin.product.vo.category.IotProductCategorySaveReqVO;
import com.pei.dehaze.module.iot.dal.dataobject.product.IotProductCategoryDO;
import com.pei.dehaze.module.iot.dal.dataobject.product.IotProductDO;
import com.pei.dehaze.module.iot.dal.mysql.product.IotProductCategoryMapper;
import com.pei.dehaze.module.iot.service.device.IotDeviceService;
import jakarta.annotation.Resource;
import org.springframework.stereotype.Service;
import org.springframework.validation.annotation.Validated;

import java.time.LocalDateTime;
import java.util.*;

import static com.pei.dehaze.framework.common.exception.util.ServiceExceptionUtil.exception;
import static com.pei.dehaze.module.iot.enums.ErrorCodeConstants.PRODUCT_CATEGORY_NOT_EXISTS;

/**
 * IoT 产品分类 Service 实现类
 *
 * @author earthyzinc
 */
@Service
@Validated
public class IotProductCategoryServiceImpl implements IotProductCategoryService {

    @Resource
    private IotProductCategoryMapper productCategoryMapper;

    @Resource
    private IotProductService productService;

    @Resource
    private IotDeviceService deviceService;

    public Long createProductCategory(IotProductCategorySaveReqVO createReqVO) {
        // 插入
        IotProductCategoryDO productCategory = BeanUtils.toBean(createReqVO, IotProductCategoryDO.class);
        productCategoryMapper.insert(productCategory);
        // 返回
        return productCategory.getId();
    }

    @Override
    public void updateProductCategory(IotProductCategorySaveReqVO updateReqVO) {
        // 校验存在
        validateProductCategoryExists(updateReqVO.getId());
        // 更新
        IotProductCategoryDO updateObj = BeanUtils.toBean(updateReqVO, IotProductCategoryDO.class);
        productCategoryMapper.updateById(updateObj);
    }

    @Override
    public void deleteProductCategory(Long id) {
        // 校验存在
        validateProductCategoryExists(id);
        // 删除
        productCategoryMapper.deleteById(id);
    }

    private void validateProductCategoryExists(Long id) {
        if (productCategoryMapper.selectById(id) == null) {
            throw exception(PRODUCT_CATEGORY_NOT_EXISTS);
        }
    }

    @Override
    public IotProductCategoryDO getProductCategory(Long id) {
        return productCategoryMapper.selectById(id);
    }

    @Override
    public List<IotProductCategoryDO> getProductCategoryList(Collection<Long> ids) {
        if (CollUtil.isEmpty(ids)) {
            return CollUtil.newArrayList();
        }
        return productCategoryMapper.selectBatchIds(ids);
    }

    @Override
    public PageResult<IotProductCategoryDO> getProductCategoryPage(IotProductCategoryPageReqVO pageReqVO) {
        return productCategoryMapper.selectPage(pageReqVO);
    }

    @Override
    public List<IotProductCategoryDO> getProductCategoryListByStatus(Integer status) {
        return productCategoryMapper.selectListByStatus(status);
    }

    @Override
    public Long getProductCategoryCount(LocalDateTime createTime) {
        return productCategoryMapper.selectCountByCreateTime(createTime);
    }

    @Override
    public Map<String, Integer> getProductCategoryDeviceCountMap() {
        // 1. 获取所有数据
        List<IotProductCategoryDO> categoryList = productCategoryMapper.selectList();
        List<IotProductDO> productList = productService.getProductList();
        // TODO @super：不要 list 查询，返回内存，而是查询一个 Map<productId, count>
        Map<Long, Integer> deviceCountMapByProductId = deviceService.getDeviceCountMapByProductId();

        // 2. 统计每个分类下的设备数量
        Map<String, Integer> categoryDeviceCountMap = new HashMap<>();
        for (IotProductCategoryDO category : categoryList) {
            categoryDeviceCountMap.put(category.getName(), 0);
            // TODO @super：CollectionUtils.getSumValue()，看看能不能简化下
            // 2.2 找到该分类下的所有产品,累加设备数量
            for (IotProductDO product : productList) {
                if (Objects.equals(product.getCategoryId(), category.getId())) {
                    Integer deviceCount = deviceCountMapByProductId.getOrDefault(product.getId(), 0);
                    categoryDeviceCountMap.merge(category.getName(), deviceCount, Integer::sum);
                }
            }
        }
        return categoryDeviceCountMap;
    }

}
