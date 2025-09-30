; Test reduction operations that should map to TOSA reduce operations
define float @test_sum_reduction(float* %array, i32 %size) {
entry:
  %init_sum = alloca float, align 4
  store float 0.0, float* %init_sum
  %cond = icmp sgt i32 %size, 0
  br i1 %cond, label %loop.header, label %exit

loop.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop.body ]
  %sum = phi float [ 0.0, %entry ], [ %new_sum, %loop.body ]
  %loop_cond = icmp slt i32 %i, %size
  br i1 %loop_cond, label %loop.body, label %exit

loop.body:
  %elem_ptr = getelementptr inbounds float, float* %array, i32 %i
  %elem = load float, float* %elem_ptr
  %new_sum = fadd float %sum, %elem
  %i.next = add nsw i32 %i, 1
  br label %loop.header

exit:
  %final_sum = phi float [ 0.0, %entry ], [ %sum, %loop.header ]
  ret float %final_sum
}

; Test max reduction
define float @test_max_reduction(float* %array, i32 %size) {
entry:
  %first_ptr = getelementptr inbounds float, float* %array, i32 0
  %init_max = load float, float* %first_ptr
  %cond = icmp sgt i32 %size, 1
  br i1 %cond, label %loop.header, label %exit

loop.header:
  %i = phi i32 [ 1, %entry ], [ %i.next, %loop.body ]
  %max_val = phi float [ %init_max, %entry ], [ %new_max, %loop.body ]
  %loop_cond = icmp slt i32 %i, %size
  br i1 %loop_cond, label %loop.body, label %exit

loop.body:
  %elem_ptr = getelementptr inbounds float, float* %array, i32 %i
  %elem = load float, float* %elem_ptr
  %cmp = fcmp ogt float %elem, %max_val
  %new_max = select float %cmp, float %elem, float %max_val
  %i.next = add nsw i32 %i, 1
  br label %loop.header

exit:
  %final_max = phi float [ %init_max, %entry ], [ %max_val, %loop.header ]
  ret float %final_max
}