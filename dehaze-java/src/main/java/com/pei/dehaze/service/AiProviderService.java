package com.pei.dehaze.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.dehaze.model.entity.SysAiProvider;
import com.pei.dehaze.model.form.ProviderForm;
import com.pei.dehaze.model.form.ProviderKeyForm;
import com.pei.dehaze.model.form.ProviderKeyUpdateForm;
import com.pei.dehaze.model.form.ProviderUpdateForm;
import com.pei.dehaze.model.query.ProviderPageQuery;
import com.pei.dehaze.model.vo.ProviderEnabledVO;
import com.pei.dehaze.model.vo.ProviderKeyVO;
import com.pei.dehaze.model.vo.ProviderVO;

import java.util.List;

/** AI 供应商与 API Key 管理（对齐 python ai_provider_service / ai_provider_key_service） */
public interface AiProviderService extends IService<SysAiProvider> {

    Page<ProviderVO> listProviders(ProviderPageQuery query);

    List<ProviderEnabledVO> listEnabledProviders();

    ProviderVO createProvider(ProviderForm form);

    ProviderVO updateProvider(Long providerId, ProviderUpdateForm form);

    void deleteProvider(Long providerId);

    List<ProviderKeyVO> listKeys(Long providerId);

    ProviderKeyVO createKey(Long providerId, ProviderKeyForm form);

    ProviderKeyVO updateKey(Long providerId, Long keyId, ProviderKeyUpdateForm form);

    void deleteKey(Long providerId, Long keyId);

    /** 管理员手动解除供应商熔断 */
    void closeCircuit(Long providerId);
}
