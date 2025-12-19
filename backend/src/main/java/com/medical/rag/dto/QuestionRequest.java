package com.medical.rag.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Data
public class QuestionRequest {
    @NotBlank(message = "问题不能为空")
    @Size(max = 2000, message = "问题长度不能超过2000字符")
    private String question;
    
    private String sessionId;
    private boolean useRewrite = false;
}
