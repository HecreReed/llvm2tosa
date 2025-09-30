; ModuleID = 'broadcast_add.c'
; target triple = "x86_64-pc-linux-gnu"

define void @matrix_vector_add_broadcast(i32* %A, i32* %B, i32* %C) {
entry:
  br label %outer.loop.header

outer.loop.header:
  %i.val = phi i32 [ 0, %entry ], [ %i.next, %outer.loop.latch ]
  %exit.cond.outer = icmp slt i32 %i.val, 2
  br i1 %exit.cond.outer, label %inner.loop.header, label %function.exit

inner.loop.header:
  %j.val = phi i32 [ 0, %outer.loop.header ], [ %j.next, %inner.loop.latch ]
  %exit.cond.inner = icmp slt i32 %j.val, 3
  br i1 %exit.cond.inner, label %loop.body, label %outer.loop.latch

loop.body:
  %i.mul.cols = mul nsw i32 %i.val, 3
  %matrix.idx = add nsw i32 %i.mul.cols, %j.val
  %ptr.A = getelementptr inbounds i32, i32* %A, i32 %matrix.idx
  %val.A = load i32, i32* %ptr.A
  %ptr.B = getelementptr inbounds i32, i32* %B, i32 %j.val
  %val.B = load i32, i32* %ptr.B
  %sum = add nsw i32 %val.A, %val.B
  %ptr.C = getelementptr inbounds i32, i32* %C, i32 %matrix.idx
  store i32 %sum, i32* %ptr.C
  br label %inner.loop.latch

inner.loop.latch:
  %j.next = add nsw i32 %j.val, 1
  br label %inner.loop.header

outer.loop.latch:
  %i.next = add nsw i32 %i.val, 1
  br label %outer.loop.header

function.exit:
  ret void
}