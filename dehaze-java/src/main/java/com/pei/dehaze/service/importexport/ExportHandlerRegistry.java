package com.pei.dehaze.service.importexport;

import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 导出处理器注册表
 * <p>启动时收集所有 {@link ExportHandler} Bean，按 module 索引。
 */
@Component
public class ExportHandlerRegistry {

    private final Map<String, ExportHandler> handlers = new HashMap<>();

    public ExportHandlerRegistry(List<ExportHandler> handlerList) {
        for (ExportHandler handler : handlerList) {
            ExportHandler existing = handlers.put(handler.getModule(), handler);
            if (existing != null) {
                throw new IllegalStateException(
                        "Duplicate ExportHandler for module: " + handler.getModule());
            }
        }
    }

    public ExportHandler getHandler(String module) {
        ExportHandler handler = handlers.get(module);
        if (handler == null) {
            throw new BusinessException(ResultCode.MODULE_IMPORT_NOT_SUPPORTED,
                    "模块 " + module + " 不支持导出");
        }
        return handler;
    }

    public boolean supports(String module) {
        return handlers.containsKey(module);
    }
}
