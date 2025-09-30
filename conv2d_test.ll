; ModuleID = 'conv2d.c'
; target triple = "x86_64-pc-linux-gnu"

define void @conv2d(float* %input, float* %kernel, float* %output, i32 %IH, i32 %IW, i32 %KH, i32 %KW) {
entry:
  %OH = sub i32 %IH, %KH
  %OH_final = add i32 %OH, 1
  %OW = sub i32 %IW, %KW
  %OW_final = add i32 %OW, 1
  br label %oh.loop.header

oh.loop.header:
  %oh.val = phi i32 [ 0, %entry ], [ %oh.next, %oh.loop.latch ]
  %oh.cond = icmp slt i32 %oh.val, %OH_final
  br i1 %oh.cond, label %ow.loop.header, label %exit

ow.loop.header:
  %ow.val = phi i32 [ 0, %oh.loop.header ], [ %ow.next, %ow.loop.latch ]
  %ow.cond = icmp slt i32 %ow.val, %OW_final
  br i1 %ow.cond, label %kh.loop.header, label %oh.loop.latch

kh.loop.header:
  %kh.val = phi i32 [ 0, %ow.loop.header ], [ %kh.next, %kh.loop.latch ]
  %sum.val = phi float [ 0.0, %ow.loop.header ], [ %new.sum, %kh.loop.latch ]
  %kh.cond = icmp slt i32 %kh.val, %KH
  br i1 %kh.cond, label %kw.loop.header, label %ow.loop.latch

kw.loop.header:
  %kw.val = phi i32 [ 0, %kh.loop.header ], [ %kw.next, %kw.loop.latch ]
  %kw.cond = icmp slt i32 %kw.val, %KW
  br i1 %kw.cond, label %conv.body, label %kh.loop.latch

conv.body:
  ; Calculate input indices: ih = oh + kh, iw = ow + kw
  %ih = add i32 %oh.val, %kh.val
  %iw = add i32 %ow.val, %kw.val
  
  ; Calculate input linear index: ih * IW + iw
  %ih.mul.IW = mul i32 %ih, %IW
  %input.idx = add i32 %ih.mul.IW, %iw
  %input.addr = getelementptr inbounds float, float* %input, i32 %input.idx
  %input.val = load float, float* %input.addr
  
  ; Calculate kernel linear index: kh * KW + kw
  %kh.mul.KW = mul i32 %kh.val, %KW
  %kernel.idx = add i32 %kh.mul.KW, %kw.val
  %kernel.addr = getelementptr inbounds float, float* %kernel, i32 %kernel.idx
  %kernel.val = load float, float* %kernel.addr
  
  ; Multiply and accumulate
  %mul.val = fmul float %input.val, %kernel.val
  %new.sum = fadd float %sum.val, %mul.val
  br label %kw.loop.latch

kw.loop.latch:
  %kw.next = add i32 %kw.val, 1
  br label %kw.loop.header

kh.loop.latch:
  %kh.next = add i32 %kh.val, 1
  br label %kh.loop.header

ow.loop.latch:
  ; Store convolution result: output[oh * OW + ow] = sum
  %oh.mul.OW = mul i32 %oh.val, %OW_final
  %output.idx = add i32 %oh.mul.OW, %ow.val
  %output.addr = getelementptr inbounds float, float* %output, i32 %output.idx
  store float %sum.val, float* %output.addr
  
  %ow.next = add i32 %ow.val, 1
  br label %ow.loop.header

oh.loop.latch:
  %oh.next = add i32 %oh.val, 1
  br label %oh.loop.header

exit:
  ret void
}