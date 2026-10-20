package com.pei.clarity.model.query;

import com.pei.clarity.common.base.BasePageQuery;
import io.swagger.v3.oas.annotations.media.Schema;

/** 无过滤条件的分页查询参数：仅承载 pageNum/pageSize 及边界校验（pageSize ≤ 100） */
@Schema(description = "分页查询参数")
public class PageParamQuery extends BasePageQuery {}
