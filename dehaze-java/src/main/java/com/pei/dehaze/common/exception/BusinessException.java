package com.pei.dehaze.common.exception;

import com.pei.dehaze.common.result.IResultCode;
import com.pei.dehaze.common.result.ResultCode;
import lombok.Getter;

/**
 * 自定义业务异常
 *
 * @author earthyzinc
 * @since 2022/7/31
 */
@Getter
public class BusinessException extends RuntimeException {

    private final IResultCode resultCode;

    public BusinessException(IResultCode errorCode) {
        super(errorCode.getMsg());
        this.resultCode = errorCode;
    }

    public BusinessException(IResultCode errorCode, String message) {
        super(message);
        this.resultCode = errorCode;
    }

    public BusinessException(String message){
        super(message);
        this.resultCode = ResultCode.BUSINESS_ERROR;
    }

    public BusinessException(String message, Throwable cause){
        super(message, cause);
        this.resultCode = ResultCode.BUSINESS_ERROR;
    }

    public BusinessException(Throwable cause){
        super(cause);
        this.resultCode = ResultCode.SYSTEM_EXECUTION_ERROR;
    }


}
