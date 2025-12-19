package com.medical.rag.dto;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import lombok.Data;

@Data
public class RetrievalRequest {
    @NotBlank(message = "查询不能为空")
    private String query;
    
    @Min(1) @Max(100)
    private int topK = 10;
    
    @Pattern(regexp = "^(bm25|vector|hybrid)$", message = "检索方式必须是 bm25, vector 或 hybrid")
    private String method = "hybrid";
}
