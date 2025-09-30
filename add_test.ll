define float @add_example(float %a, float %b) {
entry:
  %result = fadd float %a, %b
  ret float %result
}