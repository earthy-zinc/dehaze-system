package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.entity.SysAiModel;
import com.pei.dehaze.model.form.AiModelForm;
import com.pei.dehaze.model.form.AiModelUpdateForm;
import com.pei.dehaze.model.form.ModelPriceForm;
import com.pei.dehaze.model.form.ModelPriceUpdateForm;
import com.pei.dehaze.model.query.AiModelPageQuery;
import com.pei.dehaze.model.query.ModelPriceQuery;
import com.pei.dehaze.model.vo.AiModelVO;
import com.pei.dehaze.model.vo.ModelPriceVO;

import java.util.List;

/** AI 模型注册表与用户售价版本管理（对齐 python ai_model_service / ai_model_price_service） */
public interface AiModelService extends IService<SysAiModel> {

    /** 模型分页列表（含近 24h 调用统计） */
    Page<AiModelVO> listModels(AiModelPageQuery query);

    /** 启用模型列表（按用户 VIP 等级过滤，可按模型类型筛选） */
    List<AiModelVO> listEnabledModels(String modelType);

    AiModelVO createModel(AiModelForm form);

    AiModelVO updateModel(String modelId, AiModelUpdateForm form);

    void deleteModel(String modelId);

    Page<ModelPriceVO> listPrices(ModelPriceQuery query);

    ModelPriceVO createPrice(ModelPriceForm form);

    ModelPriceVO updatePrice(Long priceId, ModelPriceUpdateForm form);

    void deletePrice(Long priceId);
}
