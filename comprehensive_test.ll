; Comprehensive LLVM IR test covering multiple instruction types
define float @comprehensive_test(float* %input_a, float* %input_b, i32 %size) {
entry:
  ; Integer comparison
  %cond = icmp sgt i32 %size, 0
  br i1 %cond, label %process, label %exit

process:
  ; Memory allocation  
  %temp = alloca float, align 4
  
  ; Arithmetic operations
  %a_val = load float, float* %input_a
  %b_val = load float, float* %input_b
  %sum = fadd float %a_val, %b_val
  %product = fmul float %sum, %a_val
  
  ; Store result
  store float %product, float* %temp
  
  ; Load and return
  %result = load float, float* %temp
  br label %exit

exit:
  %final = phi float [ 0.0, %entry ], [ %result, %process ]
  ret float %final
}