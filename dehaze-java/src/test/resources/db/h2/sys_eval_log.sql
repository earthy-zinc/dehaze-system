create table dehaze_test.sys_eval_log
(
    id           bigint auto_increment comment 'id'
        primary key,
    algorithm_id bigint                             not null comment '算法id',
    pred_file_id bigint                             null comment '预测图像文件id',
    pred_md5     char(32)                           not null comment '预测图像md5值',
    pred_url     text                               not null comment '预测图像url',
    gt_file_id   bigint                             null comment '真值图像文件id',
    gt_md5       char(32)                           not null comment '真值图像md5值',
    gt_url       text                               not null comment '真值图像url',
    time         int      default 0                 null comment '评估时间（毫秒）',
    status       varchar(20) default 'completed'    not null comment '任务状态：processing/completed/failed',
    error_message text                              null comment '失败错误信息',
    result       json                               null comment '预测结果',
    create_time  datetime default CURRENT_TIMESTAMP not null comment '创建时间',
    update_time  datetime default CURRENT_TIMESTAMP not null on update CURRENT_TIMESTAMP comment '更新时间',
    create_by    bigint                             null comment '创建人ID',
    update_by    bigint                             null comment '修改人ID'
);

create index idx_algorithm_id
    on dehaze_test.sys_eval_log (algorithm_id);

create index idx_gt_md5
    on dehaze_test.sys_eval_log (gt_md5);

create index idx_pred_md5
    on dehaze_test.sys_eval_log (pred_md5);

create index idx_status
    on dehaze_test.sys_eval_log (status);

