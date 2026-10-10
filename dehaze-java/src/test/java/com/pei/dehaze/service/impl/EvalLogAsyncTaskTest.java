package com.pei.dehaze.service.impl;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import cn.hutool.json.JSONObject;
import com.pei.dehaze.common.enums.LogStatusEnum;
import com.pei.dehaze.mapper.SysEvalLogMapper;
import com.pei.dehaze.model.entity.SysEvalLog;
import com.pei.dehaze.service.client.PythonAlgorithmClient;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

/**
 * 评估异步落库口径单元测试。
 * <p>
 * 评估图与参考图的 MD5 都是**图片内容** MD5，由算法服务（Python）下载图片后算出并随响应返回；
 * Java 侧不得用 URL 的 MD5 冒充，也不得漏写 pred_md5 / gt_md5。
 */
@DisplayName("EvalLogAsyncTask 结果落库口径单元测试")
@ExtendWith(MockitoExtension.class)
class EvalLogAsyncTaskTest {

    private static final String MODEL = "aecrnet";
    private static final String PRED_MD5 = "d41d8cd98f00b204e9800998ecf8427e";
    private static final String GT_MD5 = "0cc175b9c0f1b6a831c399e269772661";

    @Mock
    private SysEvalLogMapper evalLogMapper;

    @Mock
    private PythonAlgorithmClient pythonClient;

    @Test
    @DisplayName("pred_md5/gt_md5 取自算法服务响应")
    void writesMd5FromAlgorithmResponse() {
        JSONObject resp = new JSONObject();
        resp.set("logId", 1);
        resp.set("status", LogStatusEnum.COMPLETED.getValue());
        resp.set("metrics", "{\"psnr\":28.5}");
        resp.set("predMd5", PRED_MD5);
        resp.set("gtMd5", GT_MD5);
        when(pythonClient.evaluate(MODEL, "http://cdn.test/pred.png", "http://cdn.test/gt.png"))
                .thenReturn(resp);

        EvalLogAsyncTask task = new EvalLogAsyncTask(evalLogMapper, pythonClient, new SimpleMeterRegistry());
        task.execute(5L, MODEL, "http://cdn.test/pred.png", "http://cdn.test/gt.png");

        ArgumentCaptor<SysEvalLog> captor = ArgumentCaptor.forClass(SysEvalLog.class);
        verify(evalLogMapper).updateById(captor.capture());
        SysEvalLog updated = captor.getValue();
        assertThat(updated.getStatus()).isEqualTo(LogStatusEnum.COMPLETED);
        assertThat(updated.getPredMd5()).isEqualTo(PRED_MD5);
        assertThat(updated.getGtMd5()).isEqualTo(GT_MD5);
    }
}
