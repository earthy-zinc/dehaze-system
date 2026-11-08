package com.pei.clarity.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.pei.clarity.model.entity.SysAiArtifact;
import java.util.List;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

/**
 * AI 中间产物访问层
 *
 * @author clarity
 */
@Mapper
public interface SysAiArtifactMapper extends BaseMapper<SysAiArtifact> {

    /**
     * 批量取消息的产物引用（消息只存引用，读模型按 messageId 分组成 {@code artifactIds}）
     */
    @Select("<script>SELECT id, message_id AS messageId FROM sys_ai_artifact WHERE message_id IN "
            + "<foreach collection='messageIds' item='id' open='(' separator=',' close=')'>#{id}</foreach> "
            + "ORDER BY id</script>")
    List<ArtifactRefRow> listRefsByMessageIds(@Param("messageIds") List<Long> messageIds);

    /**
     * 产物引用行（自定义查询投影）
     */
    class ArtifactRefRow {
        private Long messageId;
        private Long id;

        public Long getMessageId() {
            return messageId;
        }

        public void setMessageId(Long messageId) {
            this.messageId = messageId;
        }

        public Long getId() {
            return id;
        }

        public void setId(Long id) {
            this.id = id;
        }
    }
}
