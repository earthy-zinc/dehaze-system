package com.pei.dehaze.service.strategy;

/**
 * 任务取消异常
 * 运行时异常，用于中断任务执行
 */
public class TaskCancelledException extends RuntimeException {

    public TaskCancelledException(String message) {
        super(message);
    }

    public TaskCancelledException(String message, Throwable cause) {
        super(message, cause);
    }
}
