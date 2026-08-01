package com.pei.dehaze.service.impl.file;

import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.service.FileService;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * 存储后端工厂：按 storage 标识取得对应的 {@link FileService} 实现。
 * Spring 启动时自动收集所有 FileService bean（多后端共存），按 {@link FileService#getStorageType()} 索引。
 * URL 拼接的唯一入口：所有需要 url 的地方应通过本工厂取得后端，再调用 {@link FileService#getUrl(String)}。
 * {@link #getDefault()} 返回 file.type 指定的默认上传后端（新上传文件写入哪个 storage）。
 *
 * @author earthyzinc
 */
@Component
public class StorageServiceFactory {

    private final Map<String, FileService> registry;
    private final String defaultStorageType;

    public StorageServiceFactory(List<FileService> services,
                                 @Value("${file.type}") String defaultStorageType) {
        this.registry = services.stream()
                .collect(Collectors.toMap(FileService::getStorageType, Function.identity()));
        this.defaultStorageType = defaultStorageType;
    }

    /**
     * 按 storage 取得后端服务；storage 为空或未注册时抛业务异常。
     */
    public FileService get(String storage) {
        FileService service = registry.get(storage);
        if (service == null) {
            throw new BusinessException("不支持的存储后端: " + storage);
        }
        return service;
    }

    /**
     * 取得默认上传后端（由 file.type 指定），用于新上传文件写入。
     */
    public FileService getDefault() {
        return get(defaultStorageType);
    }
}
