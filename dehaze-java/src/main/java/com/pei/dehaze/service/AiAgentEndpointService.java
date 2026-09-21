package com.pei.dehaze.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import com.pei.dehaze.common.util.AiCredentialCipher;
import com.pei.dehaze.mapper.SysAiAgentEndpointMapper;
import com.pei.dehaze.model.entity.SysAiAgentEndpoint;
import com.pei.dehaze.model.form.AiEndpointCreateForm;
import com.pei.dehaze.model.form.AiEndpointUpdateForm;
import com.pei.dehaze.model.query.AiEndpointPageQuery;
import com.pei.dehaze.model.vo.AiEndpointVO;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.net.InetAddress;
import java.net.URI;
import java.net.UnknownHostException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 外部 A2A 端点管理服务（注册/更新/删除/分页）。
 *
 * <p>对齐 dehaze-python {@code ai_agent_endpoint_service}：base_url 唯一（软删行复活）、
 * 仅 https 且禁内网（SSRF 防护）、凭证仅存 AES 密文、注册/更新后尽力拉取 Agent Card。
 *
 * @author dehaze
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class AiAgentEndpointService {

    private static final HttpClient HTTP_CLIENT = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(5))
            .followRedirects(HttpClient.Redirect.NEVER)
            .build();

    /**
     * IPv4 阻断段（CIDR 表驱动）：与 dehaze-python {@code _is_internal_ip} 同口径。
     *
     * <p>表外由标准库方法兜（环回/私有/链路本地/任意/组播）；表内是标准库不覆盖的段——
     * 八位组手算在 {@code /10}、{@code /15} 这类非 8/16 边界上极易漏判，加段只需加一行。
     */
    private static final List<Cidr> BLOCKED_IPV4 = List.of(
            Cidr.of("0.0.0.0/8"),
            Cidr.of("9.0.0.0/8"),
            Cidr.of("11.0.0.0/8"),
            Cidr.of("21.0.0.0/8"),
            Cidr.of("30.0.0.0/8"),
            Cidr.of("100.64.0.0/10"),
            Cidr.of("192.0.2.0/24"),
            Cidr.of("198.18.0.0/15"),
            Cidr.of("198.51.100.0/24"),
            Cidr.of("203.0.113.0/24"),
            Cidr.of("224.0.0.0/4"),
            Cidr.of("240.0.0.0/4"));

    private record Cidr(int network, int bits) {

        static Cidr of(String literal) {
            String[] parts = literal.split("/");
            String[] octets = parts[0].split("\\.");
            int network = Integer.parseInt(octets[0]) << 24 | Integer.parseInt(octets[1]) << 16
                    | Integer.parseInt(octets[2]) << 8 | Integer.parseInt(octets[3]);
            return new Cidr(network, Integer.parseInt(parts[1]));
        }

        boolean contains(int address) {
            int mask = bits == 0 ? 0 : -1 << (32 - bits);
            return (address & mask) == (network & mask);
        }
    }

    private static int ipv4ToInt(byte[] bytes) {
        return (bytes[0] & 0xFF) << 24 | (bytes[1] & 0xFF) << 16
                | (bytes[2] & 0xFF) << 8 | (bytes[3] & 0xFF);
    }

    private final SysAiAgentEndpointMapper endpointMapper;

    private final AiCredentialCipher credentialCipher;

    private final AuditLogService auditLogService;

    private final ObjectMapper objectMapper;

    public IPage<AiEndpointVO> list(AiEndpointPageQuery query) {
        LambdaQueryWrapper<SysAiAgentEndpoint> wrapper = new LambdaQueryWrapper<SysAiAgentEndpoint>()
                .eq(query.getStatus() != null, SysAiAgentEndpoint::getStatus, query.getStatus())
                .orderByDesc(SysAiAgentEndpoint::getId);
        if (query.getKeyword() != null && !query.getKeyword().isBlank()) {
            String keyword = query.getKeyword();
            wrapper.and(w -> w.like(SysAiAgentEndpoint::getName, keyword)
                    .or().like(SysAiAgentEndpoint::getBaseUrl, keyword));
        }
        Page<SysAiAgentEndpoint> page = new Page<>(query.getPageNum(), query.getPageSize());
        IPage<SysAiAgentEndpoint> endpointPage = endpointMapper.selectPage(page, wrapper);
        Page<AiEndpointVO> result = new Page<>(query.getPageNum(), query.getPageSize(), endpointPage.getTotal());
        result.setRecords(new ArrayList<>(endpointPage.getRecords().stream().map(this::toVO).toList()));
        return result;
    }

    @Transactional
    public AiEndpointVO create(AiEndpointCreateForm form) {
        String baseUrl = form.getBaseUrl().replaceAll("/+$", "");
        if (!isSafeUrl(baseUrl) || (form.getAgentCardUrl() != null && !isSafeUrl(form.getAgentCardUrl()))) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "base_url/agent_card_url 仅支持 https 且禁止内网地址");
        }
        String credential = form.getCredential() == null || form.getCredential().isBlank()
                ? null : credentialCipher.encrypt(form.getCredential());
        String authType = form.getAuthType() == null ? "http" : form.getAuthType();
        Integer status = form.getStatus() == null ? 1 : form.getStatus();
        SysAiAgentEndpoint existing = endpointMapper.selectByBaseUrlIgnoringDeleted(baseUrl);
        SysAiAgentEndpoint endpoint;
        if (existing != null) {
            if (existing.getDeleted() == null || existing.getDeleted() == 0) {
                throw new BusinessException(ResultCode.DATA_EXISTS, "该端点地址已注册");
            }
            // 软删行占用唯一键 uk_base_url(base_url, deleted)，复活原行并覆盖为新表单值
            existing.setName(form.getName());
            existing.setAgentCardUrl(form.getAgentCardUrl());
            existing.setAuthType(authType);
            existing.setCredential(credential);
            existing.setStatus(status);
            existing.setDeleted(0L);
            endpointMapper.updateById(existing);
            endpoint = existing;
        } else {
            endpoint = new SysAiAgentEndpoint();
            endpoint.setName(form.getName());
            endpoint.setAgentCardUrl(form.getAgentCardUrl());
            endpoint.setBaseUrl(baseUrl);
            endpoint.setAuthType(authType);
            endpoint.setCredential(credential);
            endpoint.setStatus(status);
            endpointMapper.insert(endpoint);
        }
        refreshAgentCardQuietly(endpoint);
        return toVO(endpoint);
    }

    @Transactional
    public void delete(Long endpointId, Long operatorId) {
        SysAiAgentEndpoint endpoint = getOrThrow(endpointId);
        endpointMapper.softDeleteByIds(List.of(endpointId));
        Map<String, Object> beforeValue = new LinkedHashMap<>();
        beforeValue.put("name", endpoint.getName());
        beforeValue.put("base_url", endpoint.getBaseUrl());
        auditLogService.recordAudit(operatorId, "ai_agent_endpoint", endpointId, "delete", "ai_agent",
                beforeValue, null, null, null);
    }

    @Transactional
    public AiEndpointVO update(Long endpointId, AiEndpointUpdateForm form) {
        SysAiAgentEndpoint endpoint = getOrThrow(endpointId);
        if (form.getBaseUrl() != null) {
            String baseUrl = form.getBaseUrl().replaceAll("/+$", "");
            if (!isSafeUrl(baseUrl)) {
                throw new BusinessException(ResultCode.PARAM_ERROR, "base_url 仅支持 https 且禁止内网地址");
            }
            endpoint.setBaseUrl(baseUrl);
        }
        if (form.getAgentCardUrl() != null) {
            if (!isSafeUrl(form.getAgentCardUrl())) {
                throw new BusinessException(ResultCode.PARAM_ERROR, "agent_card_url 仅支持 https 且禁止内网地址");
            }
            endpoint.setAgentCardUrl(form.getAgentCardUrl());
        }
        if (form.getName() != null) {
            endpoint.setName(form.getName());
        }
        if (form.getAuthType() != null) {
            endpoint.setAuthType(form.getAuthType());
        }
        if (form.getCredential() != null && !form.getCredential().isBlank()) {
            endpoint.setCredential(credentialCipher.encrypt(form.getCredential()));
        }
        if (form.getStatus() != null) {
            endpoint.setStatus(form.getStatus());
        }
        endpointMapper.updateById(endpoint);
        refreshAgentCardQuietly(endpoint);
        return toVO(endpoint);
    }

    /**
     * 拉取并缓存 Agent Card；拉取失败仅告警（不阻断端点注册/更新）
     */
    private void refreshAgentCardQuietly(SysAiAgentEndpoint endpoint) {
        if (endpoint.getAgentCardUrl() == null || !isSafeUrl(endpoint.getAgentCardUrl())) {
            return;
        }
        try {
            HttpRequest request = HttpRequest.newBuilder(URI.create(endpoint.getAgentCardUrl()))
                    .timeout(Duration.ofSeconds(5))
                    .GET()
                    .build();
            HttpResponse<String> response = HTTP_CLIENT.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() >= 200 && response.statusCode() < 300) {
                endpoint.setAgentCard(objectMapper.readValue(response.body(), Map.class));
                endpointMapper.updateById(endpoint);
            } else {
                log.warn("端点 {} Agent Card 拉取失败: HTTP {}", endpoint.getId(), response.statusCode());
            }
        } catch (Exception e) {
            log.warn("端点 {} Agent Card 拉取失败: {}", endpoint.getId(), e.getMessage());
        }
    }

    /**
     * SSRF 防护：仅 https，且主机（含域名解析出的全部地址）不得是环回/内网/链路本地/保留地址。
     *
     * <p>对齐 dehaze-python {@code utils/ssrf.is_safe_url}：域名按全部解析结果判定（防 DNS 重绑定）、
     * 解析失败保守拒绝。IPv6 字面量由 {@link InetAddress} 统一判定——{@code URI.getHost()} 返回的是
     * 带方括号形式（如 {@code [::1]}），按字符串前缀判定会漏掉整个 IPv6 内网段，必须先剥离方括号。
     */
    static boolean isSafeUrl(String url) {
        if (url == null || url.isBlank()) {
            return false;
        }
        URI uri;
        try {
            uri = URI.create(url);
        } catch (Exception e) {
            return false;
        }
        if (!"https".equalsIgnoreCase(uri.getScheme()) || uri.getHost() == null) {
            return false;
        }
        String host = uri.getHost().toLowerCase();
        if (host.startsWith("[") && host.endsWith("]")) {
            host = host.substring(1, host.length() - 1);
        }
        if ("localhost".equals(host) || host.endsWith(".local") || host.endsWith(".internal")) {
            return false;
        }
        try {
            for (InetAddress address : InetAddress.getAllByName(host)) {
                if (isInternalAddress(address)) {
                    return false;
                }
            }
        } catch (UnknownHostException e) {
            return false;
        }
        return true;
    }

    private static boolean isInternalAddress(InetAddress address) {
        if (address.isAnyLocalAddress() || address.isLoopbackAddress() || address.isLinkLocalAddress()
                || address.isSiteLocalAddress() || address.isMulticastAddress()) {
            return true;
        }
        byte[] bytes = address.getAddress();
        if (bytes.length == 4) {
            int value = ipv4ToInt(bytes);
            return BLOCKED_IPV4.stream().anyMatch(cidr -> cidr.contains(value));
        }
        if ((bytes[0] & 0xFE) == 0xFC) {
            // ULA fc00::/7：JDK 的 isSiteLocalAddress 只覆盖已废弃的 fec0::/10
            return true;
        }
        for (int i = 0; i < 10; i++) {
            if (bytes[i] != 0) {
                return false;
            }
        }
        // IPv4 映射地址（::ffff:a.b.c.d）：JDK 通常已规范化为 Inet4Address 并走上面的 IPv4 判定，
        // 此处仅覆盖未规范化的兜底情形；Python 同样按映射到的 IPv4 判定（::ffff:8.8.8.8 放行、::ffff:127.0.0.1 拒绝）
        return bytes[10] == (byte) 0xFF && bytes[11] == (byte) 0xFF;
    }

    private SysAiAgentEndpoint getOrThrow(Long endpointId) {
        SysAiAgentEndpoint endpoint = endpointMapper.selectById(endpointId);
        if (endpoint == null) {
            throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "端点不存在");
        }
        return endpoint;
    }

    private AiEndpointVO toVO(SysAiAgentEndpoint endpoint) {
        AiEndpointVO vo = new AiEndpointVO();
        vo.setId(endpoint.getId());
        vo.setName(endpoint.getName());
        vo.setAgentCardUrl(endpoint.getAgentCardUrl());
        vo.setBaseUrl(endpoint.getBaseUrl());
        vo.setAuthType(endpoint.getAuthType());
        vo.setAgentCard(endpoint.getAgentCard());
        vo.setStatus(endpoint.getStatus());
        vo.setCreateTime(endpoint.getCreateTime());
        return vo;
    }
}
