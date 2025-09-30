; Test various LLVM instructions that should map to different TOSA operators
define void @test_conv_ops() {
entry:
  ret void
}

; Test arithmetic operations -> tosa.add, tosa.sub, tosa.mul, etc.
define float @test_arithmetic(float %a, float %b) {
entry:
  %add_result = fadd float %a, %b
  %sub_result = fsub float %add_result, %b
  %mul_result = fmul float %sub_result, %a
  %div_result = fdiv float %mul_result, %b
  ret float %div_result
}

; Test comparison operations -> tosa.equal, tosa.greater, tosa.greater_equal
define i1 @test_comparisons(i32 %a, i32 %b) {
entry:
  %eq = icmp eq i32 %a, %b
  %gt = icmp sgt i32 %a, %b
  %ge = icmp sge i32 %a, %b
  %result = and i1 %eq, %gt
  ret i1 %result
}

; Test math functions -> tosa.abs, tosa.exp, tosa.log, tosa.sin, tosa.cos
define float @test_math_functions(float %x) {
entry:
  %abs_result = call float @llvm.fabs.f32(float %x)
  %exp_result = call float @llvm.exp.f32(float %abs_result)
  %log_result = call float @llvm.log.f32(float %exp_result)
  %sin_result = call float @llvm.sin.f32(float %log_result)
  %cos_result = call float @llvm.cos.f32(float %sin_result)
  ret float %cos_result
}

; Test control flow -> tosa.cond_if, tosa.select
define float @test_control_flow(i1 %cond, float %a, float %b) {
entry:
  br i1 %cond, label %true_branch, label %false_branch

true_branch:
  %true_result = fmul float %a, 2.0
  br label %merge

false_branch:
  %false_result = fmul float %b, 3.0
  br label %merge

merge:
  %result = phi float [ %true_result, %true_branch ], [ %false_result, %false_branch ]
  ret float %result
}

; Test memory operations -> tosa tensor operations
define float @test_memory_ops(float* %ptr) {
entry:
  %alloc = alloca float, align 4
  %loaded = load float, float* %ptr
  store float %loaded, float* %alloc
  %result = load float, float* %alloc
  ret float %result
}

; Test vector operations -> tosa data layout operations
define <4 x float> @test_vector_ops(<4 x float> %v1, <4 x float> %v2) {
entry:
  %elem = extractelement <4 x float> %v1, i32 0
  %inserted = insertelement <4 x float> %v2, float %elem, i32 1
  ret <4 x float> %inserted
}

declare float @llvm.fabs.f32(float)
declare float @llvm.exp.f32(float)
declare float @llvm.log.f32(float)
declare float @llvm.sin.f32(float)
declare float @llvm.cos.f32(float)