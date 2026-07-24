package com.pei.dehaze.model.form;

import jakarta.validation.constraints.NotBlank;
import lombok.Data;

import java.time.LocalDateTime;

@Data
public class ApiKeyForm {

    @NotBlank
    private String name;

    private LocalDateTime expiresAt;
}
