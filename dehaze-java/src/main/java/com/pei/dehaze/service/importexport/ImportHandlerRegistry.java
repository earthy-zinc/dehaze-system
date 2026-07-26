package com.pei.dehaze.service.importexport;

import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 导入处理器注册表
 */
@Component
public class ImportHandlerRegistry {

    private final Map<String, ImportHandler> handlers = new HashMap<>();

    public ImportHandlerRegistry(List<ImportHandler> handlerList) {
        for (ImportHandler handler : handlerList) {
            ImportHandler existing = handlers.put(handler.getModule(), handler);
            if (existing != null) {
                throw new IllegalStateException(
                        "Duplicate ImportHandler for module: " + handler.getModule());
            }
        }
    }

    public ImportHandler getHandler(String module) {
        ImportHandler handler = handlers.get(module);
        if (handler == null) {
            throw new BusinessException(ResultCode.MODULE_IMPORT_NOT_SUPPORTED,
                    "模块 " + module + " 不支持导入");
        }
        return handler;
    }

    public boolean supports(String module) {
        return handlers.containsKey(module);
    }
}
