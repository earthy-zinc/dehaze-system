package com.pei.clarity.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.pei.clarity.model.entity.SysAiProvider;
import com.pei.clarity.model.form.ProviderForm;
import com.pei.clarity.model.form.ProviderKeyForm;
import com.pei.clarity.model.form.ProviderKeyUpdateForm;
import com.pei.clarity.model.form.ProviderUpdateForm;
import com.pei.clarity.model.query.ProviderPageQuery;
import com.pei.clarity.model.vo.ProviderEnabledVO;
import com.pei.clarity.model.vo.ProviderKeyVO;
import com.pei.clarity.model.vo.ProviderVO;
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
