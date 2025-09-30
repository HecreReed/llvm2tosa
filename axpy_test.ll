; ModuleID = 'axpy.c'
; target triple = "x86_64-pc-linux-gnu"

define void @axpy(float %a, float* %x, float* %y, i32 %n) {
entry:
  %start.cond = icmp sgt i32 %n, 0
  br i1 %start.cond, label %loop.header, label %loop.exit

loop.header:
  %i.val = phi i32 [ 0, %entry ], [ %i.next, %loop.latch ]
  %loop.cond = icmp slt i32 %i.val, %n
  br i1 %loop.cond, label %loop.body, label %loop.exit

loop.body:
  %x.addr = getelementptr inbounds float, float* %x, i32 %i.val
  %x.val = load float, float* %x.addr
  %y.addr = getelementptr inbounds float, float* %y, i32 %i.val
  %y.val = load float, float* %y.addr
  %mul.val = fmul float %a, %x.val
  %add.val = fadd float %mul.val, %y.val
  store float %add.val, float* %y.addr
  br label %loop.latch

loop.latch:
  %i.next = add nsw i32 %i.val, 1
  br label %loop.header

loop.exit:
  ret void
}