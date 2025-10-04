package com.pei.dehaze.common.validator;

import jakarta.validation.Constraint;
import jakarta.validation.Payload;

import java.lang.annotation.*;

@Documented
@Constraint(validatedBy = FileExistValidator.class)
@Target({ElementType.FIELD})
@Retention(RetentionPolicy.RUNTIME)
public @interface FileExists {
    String message() default "指定的文件不存在于服务器";

    Class<?>[] groups() default {};

    Class<? extends Payload>[] payload() default {};
}
