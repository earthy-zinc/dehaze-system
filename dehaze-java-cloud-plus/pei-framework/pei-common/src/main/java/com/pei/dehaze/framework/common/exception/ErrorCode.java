package com.pei.dehaze.framework.common.exception;

import com.pei.dehaze.framework.common.exception.enums.GlobalErrorCodeConstants;
import com.pei.dehaze.framework.common.exception.enums.ServiceErrorCodeRange;

/**
 * 错误码对象
 * <p>
 * 全局错误码，占用 [0, 999], 参见 {@link GlobalErrorCodeConstants} 业务异常错误码，占用 [1 000 000 000, +∞)，参见
 * {@link ServiceErrorCodeRange}
 * <p>
 * TODO 错误码设计成对象的原因，为未来的 i18 国际化做准备
 *
 * @param code 错误码
 * @param msg  错误提示
 */
public record ErrorCode(Integer code, String msg) {

}
