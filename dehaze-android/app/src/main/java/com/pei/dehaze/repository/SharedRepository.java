package com.pei.dehaze.repository;

import com.pei.dehaze.sdk.api.AlgorithmAPI;
import com.pei.dehaze.sdk.api.FileAPI;
import com.pei.dehaze.sdk.api.ModelAPI;
import com.pei.dehaze.sdk.model.Option;
import com.pei.dehaze.sdk.model.evaluation.EvalParam;
import com.pei.dehaze.sdk.model.evaluation.EvalResult;
import com.pei.dehaze.sdk.model.evaluation.EvaluationLogVO;
import com.pei.dehaze.sdk.model.file.FileInfo;
import com.pei.dehaze.sdk.model.prediction.PredParam;
import com.pei.dehaze.sdk.model.prediction.PredResult;
import com.pei.dehaze.sdk.model.prediction.PredictionLogVO;

import java.io.File;
import java.util.List;

/**
 * 跨模块共享操作 Repository。
 * <p>
 * 提供 "上传图片 / 算法下拉选项 / 去雾预测 / 效果评估 / 预测日志 / 评估日志" 等多个模块共用的 SDK 调用入口，
 * 避免在 Compare / Presentation / Evaluation 等业务 ViewModel 中各自重复实现 SDK 调用样板。
 * <p>
 * 原则：本类只做 SDK 调用 + RepositoryAdapters 适配，不包含业务编排逻辑。
 */
public class SharedRepository {

    /**
     * 上传图片（去雾/评估/演示等多模块共用）
     */
    public void uploadImage(File imageFile, RepositoryCallback<FileInfo> callback) {
        FileAPI.upload(imageFile, RepositoryAdapters.wrap(callback));
    }

    /**
     * 获取算法下拉选项
     */
    public void getAlgorithmOptions(RepositoryCallback<List<Option>> callback) {
        AlgorithmAPI.getOption(RepositoryAdapters.wrap(callback));
    }

    /**
     * 提交去雾预测并轮询至终态（completed/failed）或超时
     */
    public void getPrediction(PredParam param, RepositoryCallback<PredResult> callback) {
        ModelAPI.predictAndWait(param, RepositoryAdapters.wrap(callback));
    }

    /**
     * 提交效果评估并轮询至终态（completed/failed）或超时
     */
    public void getEvaluation(EvalParam param, RepositoryCallback<EvalResult> callback) {
        ModelAPI.evaluateAndWait(param, RepositoryAdapters.wrap(callback));
    }

    /**
     * 分页查询预测日志（拆包为 List）
     */
    public void listPredictionLogs(int pageNum, int pageSize, RepositoryCallback<List<PredictionLogVO>> callback) {
        ModelAPI.listPredictionLogs(null, pageNum, pageSize, RepositoryAdapters.wrapPage(callback));
    }

    /**
     * 分页查询评估日志（拆包为 List）
     */
    public void listEvaluationLogs(int pageNum, int pageSize, RepositoryCallback<List<EvaluationLogVO>> callback) {
        ModelAPI.listEvaluationLogs(null, pageNum, pageSize, RepositoryAdapters.wrapPage(callback));
    }
}
