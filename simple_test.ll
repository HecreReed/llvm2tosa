; Simple test for basic TOSA operations
define float @test_add_simple(float %a, float %b) {
entry:
  %result = fadd float %a, %b
  ret float %result
}

define float @test_reciprocal_simple(float %input) {
entry:
  %result = fdiv float 1.0, %input
  ret float %result
}

define float @test_exp_simple(float %input) {
entry:
  %result = call float @llvm.exp.f32(float %input)
  ret float %result
}

define i1 @test_greater_simple(float %a, float %b) {
entry:
  %result = fcmp ogt float %a, %b
  ret i1 %result
}

declare float @llvm.exp.f32(float)