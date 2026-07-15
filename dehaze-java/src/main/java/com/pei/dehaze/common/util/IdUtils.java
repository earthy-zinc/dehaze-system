package com.pei.dehaze.common.util;

import com.pei.dehaze.common.exception.BusinessException;
import com.pei.dehaze.common.result.ResultCode;

import java.util.Arrays;
import java.util.List;

/**
 * ID 集合解析工具类
 *
 * @author earthyzinc
 */
public final class IdUtils {

    private IdUtils() {
    }

    /**
     * 解析逗号分隔的ID字符串为 Long 列表
     *
     * @param idsStr 逗号分隔的ID字符串
     * @return Long 列表
     */
    public static List<Long> parseIdList(String idsStr) {
        try {
            return Arrays.stream(idsStr.split(","))
                    .map(Long::parseLong)
                    .toList();
        } catch (NumberFormatException e) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "ID格式错误");
        }
    }
}
