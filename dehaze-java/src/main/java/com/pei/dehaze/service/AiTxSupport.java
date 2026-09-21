package com.pei.dehaze.service;

import org.springframework.transaction.support.TransactionSynchronization;
import org.springframework.transaction.support.TransactionSynchronizationManager;

/**
 * AI 域事务提交后回调助手。
 *
 * <p>缓存失效与审计留痕必须在事务提交后执行：提交前失效会被并发请求击穿回填旧值，
 * 回滚后留痕会误导事后追溯。
 *
 * @author dehaze
 */
public final class AiTxSupport {

    private AiTxSupport() {
    }

    public static void afterCommit(Runnable action) {
        if (TransactionSynchronizationManager.isSynchronizationActive()) {
            TransactionSynchronizationManager.registerSynchronization(new TransactionSynchronization() {
                @Override
                public void afterCommit() {
                    action.run();
                }
            });
        } else {
            action.run();
        }
    }
}
