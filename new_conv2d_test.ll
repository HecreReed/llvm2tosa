; ModuleID = 'conv2d.c'
define void @conv2d(float* %input, float* %kernel, float* %output, 
                   i32 %IH, i32 %IW, i32 %KH, i32 %KW) {
entry:
  %OH = sub i32 %IH, %KH
  %OH_plus1 = add i32 %OH, 1
  %OW = sub i32 %IW, %KW
  %OW_plus1 = add i32 %OW, 1
  br label %oh.loop.header

oh.loop.header:
  %oh.val = phi i32 [ 0, %entry ], [ %oh.next, %oh.loop.latch ]
  %oh.cond = icmp slt i32 %oh.val, %OH_plus1
  br i1 %oh.cond, label %ow.loop.header, label %exit

ow.loop.header:
  %ow.val = phi i32 [ 0, %oh.loop.header ], [ %ow.next, %ow.loop.latch ]
  %ow.cond = icmp slt i32 %ow.val, %OW_plus1
  br i1 %ow.cond, label %kh.loop.header, label %oh.loop.latch

kh.loop.header:
  %kh.val = phi i32 [ 0, %ow.loop.header ], [ %kh.next, %kh.loop.latch ]
  %sum.val = phi float [ 0.0, %ow.loop.header ], [ %new.sum, %kh.loop.latch ]
  %kh.cond = icmp slt i32 %kh.val, %KH
  br i1 %kh.cond, label %kw.loop.header, label %ow.loop.latch

kw.loop.header:
  %kw.val = phi i32 [ 0, %kh.loop.header ], [ %kw.next, %kw.loop.latch ]
  %inner.sum.val = phi float [ %sum.val, %kh.loop.header ], [ %new.sum, %kw.loop.latch ]
  %kw.cond = icmp slt i32 %kw.val, %KW
  br i1 %kw.cond, label %loop.body, label %kh.loop.latch

loop.body:
  %ih = add i32 %oh.val, %kh.val
  %iw = add i32 %ow.val, %kw.val
  
  %idx.input.mul = mul i32 %ih, %IW
  %idx.input = add i32 %idx.input.mul, %iw
  %input.addr = getelementptr float, float* %input, i32 %idx.input
  %input.val = load float, float* %input.addr
  
  %idx.kernel.mul = mul i32 %kh.val, %KW
  %idx.kernel = add i32 %idx.kernel.mul, %kw.val
  %kernel.addr = getelementptr float, float* %kernel, i32 %idx.kernel
  %kernel.val = load float, float* %kernel.addr
  
  %mul = fmul float %input.val, %kernel.val
  %new.sum = fadd float %inner.sum.val, %mul
  br label %kw.loop.latch

kw.loop.latch:
  %kw.next = add i32 %kw.val, 1
  br label %kw.loop.header

kh.loop.latch:
  %kh.next = add i32 %kh.val, 1
  br label %kh.loop.header

ow.loop.latch:
  %idx.output.mul = mul i32 %oh.val, %OW_plus1
  %idx.output = add i32 %idx.output.mul, %ow.val
  %output.addr = getelementptr float, float* %output, i32 %idx.output
  store float %sum.val, float* %output.addr
  
  %ow.next = add i32 %ow.val, 1
  br label %ow.loop.header

oh.loop.latch:
  %oh.next = add i32 %oh.val, 1
  br label %oh.loop.header

exit:
  ret void
}