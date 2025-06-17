package com.pei.dehaze.module.promotion.api.combination;

import com.pei.dehaze.framework.common.pojo.CommonResult;
import com.pei.dehaze.framework.common.util.object.BeanUtils;
import com.pei.dehaze.module.promotion.api.combination.dto.CombinationRecordCreateReqDTO;
import com.pei.dehaze.module.promotion.api.combination.dto.CombinationRecordCreateRespDTO;
import com.pei.dehaze.module.promotion.api.combination.dto.CombinationRecordRespDTO;
import com.pei.dehaze.module.promotion.api.combination.dto.CombinationValidateJoinRespDTO;
import com.pei.dehaze.module.promotion.convert.combination.CombinationActivityConvert;
import com.pei.dehaze.module.promotion.dal.dataobject.combination.CombinationRecordDO;
import com.pei.dehaze.module.promotion.service.combination.CombinationRecordService;
import jakarta.annotation.Resource;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.RestController;

import static com.pei.dehaze.framework.common.pojo.CommonResult.success;

/**
 * 拼团活动 API 实现类
 *
 * @author HUIHUI
 */
@RestController // 提供 RESTful API 接口，给 Feign 调用
@Validated
public class CombinationRecordApiImpl implements CombinationRecordApi {

    @Resource
    private CombinationRecordService combinationRecordService;

    @Override
    public CommonResult<Boolean> validateCombinationRecord(Long userId, Long activityId, Long headId, Long skuId, Integer count) {
        combinationRecordService.validateCombinationRecord(userId, activityId, headId, skuId, count);
        return success(true);
    }

    @Override
    public CommonResult<CombinationRecordCreateRespDTO> createCombinationRecord(CombinationRecordCreateReqDTO reqDTO) {
        return success(CombinationActivityConvert.INSTANCE.convert4(combinationRecordService.createCombinationRecord(reqDTO)));
    }

    @Override
    public CommonResult<CombinationRecordRespDTO> getCombinationRecordByOrderId(Long userId, Long orderId) {
        CombinationRecordDO record = combinationRecordService.getCombinationRecord(userId, orderId);
        return success(BeanUtils.toBean(record, CombinationRecordRespDTO.class));
    }

    @Override
    public CommonResult<CombinationValidateJoinRespDTO> validateJoinCombination(
            Long userId, Long activityId, Long headId, Long skuId, Integer count) {
        return success(combinationRecordService.validateJoinCombination(userId, activityId, headId, skuId, count));
    }

}
