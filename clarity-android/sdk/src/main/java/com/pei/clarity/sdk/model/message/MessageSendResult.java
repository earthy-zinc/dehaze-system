package com.pei.clarity.sdk.model.message;

import lombok.Data;

import java.util.List;

@Data
public class MessageSendResult {
    private List<Long> messageIds;
}
