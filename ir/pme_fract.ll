; ModuleID = 'pme_spline_and_spread_hot.bc'
source_filename = "ld-temp.o"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

%llvm.amdgcn.kernel._Z24pmeSplineAndSpreadKernelILi4ELb1ELb1ELb1ELb1ELi1ELb0EL14ThreadsPerAtom0ELi32EEv18PmeGpuKernelParams.lds.t.2 = type { [384 x float], [96 x i32] }
%struct.PmeGpuKernelParams = type { %struct.PmeGpuConstParams, %struct.PmeGpuGridParams, %struct.PmeGpuAtomParams, %struct.PmeGpuDynamicParams, i8, i32, i32, i32, ptr, ptr, ptr, i64, i32, i32, ptr, ptr }
%struct.PmeGpuConstParams = type { float, [2 x ptr] }
%struct.PmeGpuGridParams = type { float, [3 x i32], [3 x float], [3 x i32], [3 x i32], [3 x i32], [3 x i32], [3 x i32], [3 x i32], [3 x i32], [3 x i32], [2 x ptr], [2 x ptr], [2 x ptr], ptr, ptr }
%struct.PmeGpuAtomParams = type { i32, ptr, [2 x ptr], ptr, ptr, ptr, ptr }
%struct.PmeGpuDynamicParams = type { [3 x [3 x float]], float, float }

$_Z24pmeSplineAndSpreadKernelILi4ELb1ELb1ELb1ELb1ELi1ELb0EL14ThreadsPerAtom0ELi32EEv18PmeGpuKernelParams = comdat any

@llvm.amdgcn.kernel._Z24pmeSplineAndSpreadKernelILi4ELb1ELb1ELb1ELb1ELi1ELb0EL14ThreadsPerAtom0ELi32EEv18PmeGpuKernelParams.lds = external hidden addrspace(3) global %llvm.amdgcn.kernel._Z24pmeSplineAndSpreadKernelILi4ELb1ELb1ELb1ELb1ELi1ELb0EL14ThreadsPerAtom0ELi32EEv18PmeGpuKernelParams.lds.t.2, align 16, !absolute_symbol !0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef align 4 ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.z() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.y() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #1

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wave.barrier() #2

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn
define protected amdgpu_kernel void @_Z24pmeSplineAndSpreadKernelILi4ELb1ELb1ELb1ELb1ELi1ELb0EL14ThreadsPerAtom0ELi32EEv18PmeGpuKernelParams(ptr addrspace(4) noundef readonly byref(%struct.PmeGpuKernelParams) align 8 captures(none) %0) local_unnamed_addr #3 comdat {
entry:
  %coerce.sroa.3.0..sroa_idx = getelementptr inbounds nuw i8, ptr addrspace(4) %0, i64 216
  %coerce.sroa.3.0.copyload = load i32, ptr addrspace(4) %coerce.sroa.3.0..sroa_idx, align 8, !amdgpu.noclobber !16
  %coerce.sroa.7.sroa.3.0.coerce.sroa.7.0..sroa_idx.sroa_idx = getelementptr inbounds nuw i8, ptr addrspace(4) %0, i64 280
  %coerce.sroa.7.sroa.3.0.copyload = load float, ptr addrspace(4) %coerce.sroa.7.sroa.3.0.coerce.sroa.7.0..sroa_idx.sroa_idx, align 8, !amdgpu.noclobber !16
  %coerce.sroa.7.sroa.4.0.coerce.sroa.7.0..sroa_idx.sroa_idx = getelementptr inbounds nuw i8, ptr addrspace(4) %0, i64 284
  %coerce.sroa.7.sroa.4.0.copyload = load float, ptr addrspace(4) %coerce.sroa.7.sroa.4.0.coerce.sroa.7.0..sroa_idx.sroa_idx, align 4, !amdgpu.noclobber !16
  %coerce.sroa.7.sroa.5.0.coerce.sroa.7.0..sroa_idx.sroa_idx = getelementptr inbounds nuw i8, ptr addrspace(4) %0, i64 288
  %coerce.sroa.7.sroa.5.0.copyload = load float, ptr addrspace(4) %coerce.sroa.7.sroa.5.0.coerce.sroa.7.0..sroa_idx.sroa_idx, align 8, !amdgpu.noclobber !16
  %coerce.sroa.7.sroa.7.0.coerce.sroa.7.0..sroa_idx.sroa_idx = getelementptr inbounds nuw i8, ptr addrspace(4) %0, i64 296
  %coerce.sroa.7.sroa.7.0.copyload = load float, ptr addrspace(4) %coerce.sroa.7.sroa.7.0.coerce.sroa.7.0..sroa_idx.sroa_idx, align 8, !amdgpu.noclobber !16
  %coerce.sroa.7.sroa.8.0.coerce.sroa.7.0..sroa_idx.sroa_idx = getelementptr inbounds nuw i8, ptr addrspace(4) %0, i64 300
  %coerce.sroa.7.sroa.8.0.copyload = load float, ptr addrspace(4) %coerce.sroa.7.sroa.8.0.coerce.sroa.7.0..sroa_idx.sroa_idx, align 4, !amdgpu.noclobber !16
  %coerce.sroa.7.sroa.10.0.coerce.sroa.7.0..sroa_idx.sroa_idx = getelementptr inbounds nuw i8, ptr addrspace(4) %0, i64 312
  %coerce.sroa.7.sroa.10.0.copyload = load float, ptr addrspace(4) %coerce.sroa.7.sroa.10.0.coerce.sroa.7.0..sroa_idx.sroa_idx, align 8, !amdgpu.noclobber !16
  %coerce.sroa.720.0..sroa_idx = getelementptr inbounds nuw i8, ptr addrspace(4) %0, i64 324
  %coerce.sroa.720.0.copyload = load i8, ptr addrspace(4) %coerce.sroa.720.0..sroa_idx, align 4, !amdgpu.noclobber !16
  %coerce.sroa.825.0..sroa_idx = getelementptr inbounds nuw i8, ptr addrspace(4) %0, i64 328
  %coerce.sroa.825.0.copyload = load i32, ptr addrspace(4) %coerce.sroa.825.0..sroa_idx, align 8, !amdgpu.noclobber !16
  %coerce.sroa.9.0..sroa_idx = getelementptr inbounds nuw i8, ptr addrspace(4) %0, i64 332
  %coerce.sroa.9.0.copyload = load i32, ptr addrspace(4) %coerce.sroa.9.0..sroa_idx, align 4, !amdgpu.noclobber !16
  %1 = tail call noundef i32 @llvm.amdgcn.workgroup.id.y()
  %2 = tail call dereferenceable(256) ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %3 = load i32, ptr addrspace(4) %2, align 4, !tbaa !7
  %mul = mul i32 %3, %1
  %4 = tail call noundef i32 @llvm.amdgcn.workgroup.id.x()
  %add = add i32 %mul, %4
  %mul3 = shl nsw i32 %add, 5
  %add4 = add nsw i32 %mul3, %coerce.sroa.825.0.copyload
  %5 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.z()
  %6 = getelementptr inbounds nuw i8, ptr addrspace(4) %2, i64 12
  %.in.i.i.i = load i16, ptr addrspace(4) %6, align 4, !tbaa !17, !range !19
  %conv.i.i = zext nneg i16 %.in.i.i.i to i32
  %7 = getelementptr inbounds nuw i8, ptr addrspace(4) %2, i64 14
  %.in.i.i.i32 = load i16, ptr addrspace(4) %7, align 2, !tbaa !17, !range !19
  %conv.i.i33 = zext nneg i16 %.in.i.i.i32 to i32
  %mul8 = mul nuw nsw i32 %5, %conv.i.i
  %mul9 = mul nuw nsw i32 %mul8, %conv.i.i33
  %8 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.y()
  %mul12 = mul nuw nsw i32 %8, %conv.i.i
  %9 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %add13 = add nuw nsw i32 %mul12, %9
  %add15 = add nuw nsw i32 %add13, %mul9
  %rem = and i32 %5, 7
  %10 = lshr i32 %add15, 2
  %mul17 = and i32 %10, 536870904
  %add18 = or disjoint i32 %mul17, %rem
  %add19 = add nsw i32 %add18, %add4
  %cmp.not = icmp slt i32 %add4, %coerce.sroa.3.0.copyload
  br i1 %cmp.not, label %if.end, label %cleanup

if.end:                                           ; preds = %entry
  %coerce.sroa.6.0..sroa_idx = getelementptr inbounds nuw i8, ptr addrspace(4) %0, i64 232
  %coerce.sroa.6.0.copyload = load ptr, ptr addrspace(4) %coerce.sroa.6.0..sroa_idx, align 8, !amdgpu.noclobber !16
  %11 = addrspacecast ptr %coerce.sroa.6.0.copyload to ptr addrspace(1)
  %coerce.sroa.513.0..sroa_idx = getelementptr inbounds nuw i8, ptr addrspace(4) %0, i64 224
  %coerce.sroa.513.0.copyload = load ptr, ptr addrspace(4) %coerce.sroa.513.0..sroa_idx, align 8, !amdgpu.noclobber !16
  %12 = addrspacecast ptr %coerce.sroa.513.0.copyload to ptr addrspace(1)
  %idxprom = sext i32 %add19 to i64
  %arrayidx21 = getelementptr inbounds [4 x i8], ptr addrspace(1) %11, i64 %idxprom
  %13 = load float, ptr addrspace(1) %arrayidx21, align 4, !tbaa !20, !amdgpu.noclobber !16
  %arrayidx25 = getelementptr inbounds [12 x i8], ptr addrspace(1) %12, i64 %idxprom
  %atomX.sroa.0.0.copyload = load float, ptr addrspace(1) %arrayidx25, align 4, !amdgpu.noclobber !16
  %atomX.sroa.4.0.arrayidx25.sroa_idx = getelementptr inbounds nuw i8, ptr addrspace(1) %arrayidx25, i64 4
  %atomX.sroa.4.0.copyload = load float, ptr addrspace(1) %atomX.sroa.4.0.arrayidx25.sroa_idx, align 4, !amdgpu.noclobber !16
  %atomX.sroa.5.0.arrayidx25.sroa_idx = getelementptr inbounds nuw i8, ptr addrspace(1) %arrayidx25, i64 8
  %atomX.sroa.5.0.copyload = load float, ptr addrspace(1) %atomX.sroa.5.0.arrayidx25.sroa_idx, align 4, !amdgpu.noclobber !16
  %agg.tmp.sroa.1.0..sroa_idx = getelementptr inbounds nuw i8, ptr addrspace(4) %0, i64 40
  %agg.tmp.sroa.1.0.copyload = load float, ptr addrspace(4) %agg.tmp.sroa.1.0..sroa_idx, align 8, !amdgpu.noclobber !16
  %agg.tmp.sroa.2.0..sroa_idx = getelementptr inbounds nuw i8, ptr addrspace(4) %0, i64 44
  %agg.tmp.sroa.2.0.copyload = load float, ptr addrspace(4) %agg.tmp.sroa.2.0..sroa_idx, align 4, !amdgpu.noclobber !16
  %agg.tmp.sroa.3.0..sroa_idx = getelementptr inbounds nuw i8, ptr addrspace(4) %0, i64 48
  %agg.tmp.sroa.3.0.copyload = load float, ptr addrspace(4) %agg.tmp.sroa.3.0..sroa_idx, align 8, !tbaa !22, !amdgpu.noclobber !16
  %agg.tmp.sroa.448.0..sroa_idx = getelementptr inbounds nuw i8, ptr addrspace(4) %0, i64 124
  %agg.tmp.sroa.448.0.copyload = load i32, ptr addrspace(4) %agg.tmp.sroa.448.0..sroa_idx, align 4, !amdgpu.noclobber !16
  %agg.tmp.sroa.5.0..sroa_idx = getelementptr inbounds nuw i8, ptr addrspace(4) %0, i64 128
  %agg.tmp.sroa.5.0.copyload = load i32, ptr addrspace(4) %agg.tmp.sroa.5.0..sroa_idx, align 8, !amdgpu.noclobber !16
  %agg.tmp.sroa.6.0..sroa_idx = getelementptr inbounds nuw i8, ptr addrspace(4) %0, i64 132
  %agg.tmp.sroa.6.0.copyload = load i32, ptr addrspace(4) %agg.tmp.sroa.6.0..sroa_idx, align 4, !tbaa !22, !amdgpu.noclobber !16
  %agg.tmp.sroa.749.0..sroa_idx = getelementptr inbounds nuw i8, ptr addrspace(4) %0, i64 200
  %agg.tmp.sroa.749.0.copyload = load ptr, ptr addrspace(4) %agg.tmp.sroa.749.0..sroa_idx, align 8, !tbaa !23, !amdgpu.noclobber !16
  %14 = addrspacecast ptr %agg.tmp.sroa.749.0.copyload to ptr addrspace(1)
  %agg.tmp.sroa.8.0..sroa_idx = getelementptr inbounds nuw i8, ptr addrspace(4) %0, i64 208
  %agg.tmp.sroa.8.0.copyload = load ptr, ptr addrspace(4) %agg.tmp.sroa.8.0..sroa_idx, align 8, !tbaa !26, !amdgpu.noclobber !16
  %15 = addrspacecast ptr %agg.tmp.sroa.8.0.copyload to ptr addrspace(1)
  %div1617.i = lshr i32 %add15, 5
  %mul13.i = shl nuw nsw i32 %div1617.i, 3
  %add14.i = or disjoint i32 %mul13.i, %rem
  %rem21.i = urem i32 %add13, 3
  %cmp24.i = icmp samesign ult i32 %add13, 3
  br i1 %cmp24.i, label %if.then26.i, label %_ZL17calculate_splinesILi4ELi32ELi8ELb0ELb0ELi1ELi32EEv18PmeGpuKernelParamsi15HIP_vector_typeIfLj3EEfPfS3_Pi.exit

if.then26.i:                                      ; preds = %if.end
  switch i32 %rem21.i, label %default.unreachable [
    i32 0, label %sw.bb.i
    i32 1, label %sw.bb46.i
    i32 2, label %sw.bb68.i
  ]

sw.bb.i:                                          ; preds = %if.then26.i
  %mul31.i = fmul fast float %atomX.sroa.0.0.copyload, %coerce.sroa.7.sroa.3.0.copyload
  %mul37.i = fmul fast float %atomX.sroa.4.0.copyload, %coerce.sroa.7.sroa.4.0.copyload
  %add38.i = fadd fast float %mul37.i, %mul31.i
  %mul44.i = fmul fast float %atomX.sroa.5.0.copyload, %coerce.sroa.7.sroa.5.0.copyload
  %add45.i = fadd fast float %add38.i, %mul44.i
  br label %sw.epilog.i

sw.bb46.i:                                        ; preds = %if.then26.i
  %mul59.i = fmul fast float %atomX.sroa.4.0.copyload, %coerce.sroa.7.sroa.7.0.copyload
  %mul66.i = fmul fast float %atomX.sroa.5.0.copyload, %coerce.sroa.7.sroa.8.0.copyload
  %add67.i = fadd fast float %mul66.i, %mul59.i
  br label %sw.epilog.i

sw.bb68.i:                                        ; preds = %if.then26.i
  %mul81.i = fmul fast float %atomX.sroa.5.0.copyload, %coerce.sroa.7.sroa.10.0.copyload
  br label %sw.epilog.i

default.unreachable:                              ; preds = %if.then26.i
  unreachable

sw.epilog.i:                                      ; preds = %sw.bb68.i, %sw.bb46.i, %sw.bb.i
  %t.0.i = phi nsz float [ %mul81.i, %sw.bb68.i ], [ %add45.i, %sw.bb.i ], [ %add67.i, %sw.bb46.i ]
  %n.0.in.sroa.speculated.i = phi float [ %agg.tmp.sroa.3.0.copyload, %sw.bb68.i ], [ %agg.tmp.sroa.1.0.copyload, %sw.bb.i ], [ %agg.tmp.sroa.2.0.copyload, %sw.bb46.i ]
  %tableIndex.0.in.sroa.speculated.i = phi i32 [ %agg.tmp.sroa.6.0.copyload, %sw.bb68.i ], [ %agg.tmp.sroa.448.0.copyload, %sw.bb.i ], [ %agg.tmp.sroa.5.0.copyload, %sw.bb46.i ]
  %add82.i = fadd fast float %t.0.i, 2.000000e+00
  %mul83.i = fmul fast float %add82.i, %n.0.in.sroa.speculated.i
  %conv84.i = fptosi float %mul83.i to i32
  %conv85.i = sitofp i32 %conv84.i to float
  %sub.i = fsub fast float %mul83.i, %conv85.i
  %add88.i = add nsw i32 %tableIndex.0.in.sroa.speculated.i, %conv84.i
  %idx.ext.i.i = sext i32 %add88.i to i64
  %add.ptr.i.i = getelementptr inbounds [4 x i8], ptr addrspace(1) %14, i64 %idx.ext.i.i
  %add.ptr.val.i.i = load float, ptr addrspace(1) %add.ptr.i.i, align 4, !tbaa !20, !amdgpu.noclobber !16
  %add93.i = fadd fast float %sub.i, %add.ptr.val.i.i
  %add.ptr.i71.i = getelementptr inbounds [4 x i8], ptr addrspace(1) %15, i64 %idx.ext.i.i
  %add.ptr.val.i72.i = load i32, ptr addrspace(1) %add.ptr.i71.i, align 4, !tbaa !3, !amdgpu.noclobber !16
  %.idx67 = mul nuw nsw i32 %add14.i, 12
  %16 = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel._Z24pmeSplineAndSpreadKernelILi4ELb1ELb1ELb1ELb1ELi1ELb0EL14ThreadsPerAtom0ELi32EEv18PmeGpuKernelParams.lds.t.2, ptr addrspace(3) @llvm.amdgcn.kernel._Z24pmeSplineAndSpreadKernelILi4ELb1ELb1ELb1ELb1ELi1ELb0EL14ThreadsPerAtom0ELi32EEv18PmeGpuKernelParams.lds, i32 0, i32 1), i32 %.idx67
  %arrayidx97.i = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %16, i32 %rem21.i
  store i32 %add.ptr.val.i72.i, ptr addrspace(3) %arrayidx97.i, align 4, !tbaa !3, !alias.scope !28, !noalias !31
  %sub107.i = fsub fast float 1.000000e+00, %add93.i
  %17 = fmul fast float %add93.i, 5.000000e-01
  %factor.op.fmul.i = fsub fast float 1.000000e+00, %17
  %factor.op.fmul20.i = fadd fast float %17, 5.000000e-01
  %18 = fmul fast float %add93.i, %add93.i
  %mul117.i = fmul fast float %18, 5.000000e-01
  %mul132.reass.i = fmul fast float %factor.op.fmul20.i, %sub107.i
  %mul140.reass.i = fmul fast float %factor.op.fmul.i, %add93.i
  %add141.i = fadd fast float %mul132.reass.i, %mul140.reass.i
  %19 = fmul fast float %sub107.i, %sub107.i
  %mul150.i = fmul fast float %19, 5.000000e-01
  %mul158.i = fmul fast float %add93.i, f0x3EAAAAAB
  %mul160.i = fmul fast float %mul158.i, %mul117.i
  %add168.i = fadd fast float %add93.i, 1.000000e+00
  %mul173.i = fmul fast float %add141.i, %add168.i
  %sub176.i = fsub fast float 3.000000e+00, %add93.i
  %mul181.i = fmul fast float %mul117.i, %sub176.i
  %add182.i = fadd fast float %mul173.i, %mul181.i
  %mul183.i = fmul fast float %add182.i, f0x3EAAAAAB
  %add168.1.i = fadd fast float %add93.i, 2.000000e+00
  %mul173.1.i = fmul fast float %mul150.i, %add168.1.i
  %sub176.1.i = fsub fast float 2.000000e+00, %add93.i
  %mul181.1.i = fmul fast float %add141.i, %sub176.1.i
  %add182.1.i = fadd fast float %mul181.1.i, %mul173.1.i
  %mul183.1.i = fmul fast float %add182.1.i, f0x3EAAAAAB
  %mul192.i = fmul fast float %sub107.i, f0x3EAAAAAB
  %mul194.i = fmul fast float %mul192.i, %mul150.i
  %.idx18.i = mul nuw nsw i32 %div1617.i, 384
  %invariant.gep.i = getelementptr inbounds nuw i8, ptr addrspace(3) @llvm.amdgcn.kernel._Z24pmeSplineAndSpreadKernelILi4ELb1ELb1ELb1ELb1ELi1ELb0EL14ThreadsPerAtom0ELi32EEv18PmeGpuKernelParams.lds, i32 %.idx18.i
  %invariant.gep24.i = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %invariant.gep.i, i32 %rem
  %.idx.i = shl nuw nsw i32 %rem21.i, 5
  %gep25.i = getelementptr inbounds nuw i8, ptr addrspace(3) %invariant.gep24.i, i32 %.idx.i
  store float %mul194.i, ptr addrspace(3) %gep25.i, align 4, !tbaa !20, !alias.scope !31, !noalias !28
  %gep25.1.i = getelementptr inbounds nuw i8, ptr addrspace(3) %gep25.i, i32 96
  store float %mul183.1.i, ptr addrspace(3) %gep25.1.i, align 4, !tbaa !20, !alias.scope !31, !noalias !28
  %gep25.2.i = getelementptr inbounds nuw i8, ptr addrspace(3) %gep25.i, i32 192
  store float %mul183.i, ptr addrspace(3) %gep25.2.i, align 4, !tbaa !20, !alias.scope !31, !noalias !28
  %gep25.3.i = getelementptr inbounds nuw i8, ptr addrspace(3) %gep25.i, i32 288
  store float %mul160.i, ptr addrspace(3) %gep25.3.i, align 4, !tbaa !20, !alias.scope !31, !noalias !28
  br label %_ZL17calculate_splinesILi4ELi32ELi8ELb0ELb0ELi1ELi32EEv18PmeGpuKernelParamsi15HIP_vector_typeIfLj3EEfPfS3_Pi.exit

_ZL17calculate_splinesILi4ELi32ELi8ELb0ELb0ELi1ELi32EEv18PmeGpuKernelParamsi15HIP_vector_typeIfLj3EEfPfS3_Pi.exit: ; preds = %sw.epilog.i, %if.end
  tail call void @llvm.amdgcn.wave.barrier()
  %cmp29 = icmp slt i32 %add19, %coerce.sroa.3.0.copyload
  br i1 %cmp29, label %land.lhs.true, label %cleanup

land.lhs.true:                                    ; preds = %_ZL17calculate_splinesILi4ELi32ELi8ELb0ELb0ELi1ELi32EEv18PmeGpuKernelParamsi15HIP_vector_typeIfLj3EEfPfS3_Pi.exit
  %tobool.not = icmp eq i8 %coerce.sroa.720.0.copyload, 0
  %cmp30 = icmp slt i32 %add19, %coerce.sroa.9.0.copyload
  %or.cond = select i1 %tobool.not, i1 true, i1 %cmp30
  br i1 %or.cond, label %if.then31, label %cleanup

if.then31:                                        ; preds = %land.lhs.true
  %agg.tmp32.sroa.1.0..sroa_idx = getelementptr inbounds nuw i8, ptr addrspace(4) %0, i64 28
  %agg.tmp32.sroa.1.0.copyload = load i32, ptr addrspace(4) %agg.tmp32.sroa.1.0..sroa_idx, align 4, !amdgpu.noclobber !16
  %agg.tmp32.sroa.2.0..sroa_idx = getelementptr inbounds nuw i8, ptr addrspace(4) %0, i64 32
  %agg.tmp32.sroa.2.0.copyload = load i32, ptr addrspace(4) %agg.tmp32.sroa.2.0..sroa_idx, align 8, !amdgpu.noclobber !16
  %agg.tmp32.sroa.3.0..sroa_idx = getelementptr inbounds nuw i8, ptr addrspace(4) %0, i64 36
  %agg.tmp32.sroa.3.0.copyload = load i32, ptr addrspace(4) %agg.tmp32.sroa.3.0..sroa_idx, align 4, !tbaa !22, !amdgpu.noclobber !16
  %agg.tmp32.sroa.446.0..sroa_idx = getelementptr inbounds nuw i8, ptr addrspace(4) %0, i64 56
  %agg.tmp32.sroa.446.0.copyload = load i32, ptr addrspace(4) %agg.tmp32.sroa.446.0..sroa_idx, align 8, !amdgpu.noclobber !16
  %agg.tmp32.sroa.5.0..sroa_idx = getelementptr inbounds nuw i8, ptr addrspace(4) %0, i64 60
  %agg.tmp32.sroa.5.0.copyload = load i32, ptr addrspace(4) %agg.tmp32.sroa.5.0..sroa_idx, align 4, !tbaa !22, !amdgpu.noclobber !16
  %agg.tmp32.sroa.647.0..sroa_idx = getelementptr inbounds nuw i8, ptr addrspace(4) %0, i64 152
  %agg.tmp32.sroa.647.0.copyload = load ptr, ptr addrspace(4) %agg.tmp32.sroa.647.0..sroa_idx, align 8, !amdgpu.noclobber !16
  %20 = addrspacecast ptr %agg.tmp32.sroa.647.0.copyload to ptr addrspace(1)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !33)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !36)
  %arrayidx16.i.idx = mul nuw nsw i32 %5, 12
  %arrayidx16.i = getelementptr inbounds nuw i8, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel._Z24pmeSplineAndSpreadKernelILi4ELb1ELb1ELb1ELb1ELi1ELb0EL14ThreadsPerAtom0ELi32EEv18PmeGpuKernelParams.lds.t.2, ptr addrspace(3) @llvm.amdgcn.kernel._Z24pmeSplineAndSpreadKernelILi4ELb1ELb1ELb1ELb1ELi1ELb0EL14ThreadsPerAtom0ELi32EEv18PmeGpuKernelParams.lds, i32 0, i32 1), i32 %arrayidx16.i.idx
  %21 = load i32, ptr addrspace(3) %arrayidx16.i, align 4, !tbaa !3, !noalias !38
  %arrayidx20.i = getelementptr inbounds nuw i8, ptr addrspace(3) %arrayidx16.i, i32 4
  %22 = load i32, ptr addrspace(3) %arrayidx20.i, align 4, !tbaa !3, !noalias !38
  %arrayidx25.i = getelementptr inbounds nuw i8, ptr addrspace(3) %arrayidx16.i, i32 8
  %23 = load i32, ptr addrspace(3) %arrayidx25.i, align 4, !tbaa !3, !noalias !38
  %add27.i = add nsw i32 %23, %9
  %cmp.not.i = icmp slt i32 %add27.i, %agg.tmp32.sroa.3.0.copyload
  %sub29.i = select i1 %cmp.not.i, i32 0, i32 %agg.tmp32.sroa.3.0.copyload
  %div31.i = lshr i32 %5, 3
  %24 = mul nuw nsw i32 %div31.i, 88
  %add4.i.i = add nuw nsw i32 %24, %5
  %.idx = mul nuw nsw i32 %9, 96
  %25 = getelementptr inbounds nuw i8, ptr addrspace(3) @llvm.amdgcn.kernel._Z24pmeSplineAndSpreadKernelILi4ELb1ELb1ELb1ELb1ELi1ELb0EL14ThreadsPerAtom0ELi32EEv18PmeGpuKernelParams.lds, i32 %.idx
  %26 = getelementptr inbounds nuw i8, ptr addrspace(3) %25, i32 64
  %arrayidx33.i = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) %26, i32 %add4.i.i
  %27 = load float, ptr addrspace(3) %arrayidx33.i, align 4, !tbaa !20, !noalias !39
  %add46.i = sub i32 %add27.i, %sub29.i
  %cmp52.not.i = icmp slt i32 %21, %agg.tmp32.sroa.1.0.copyload
  %sub57.i = select i1 %cmp52.not.i, i32 0, i32 %agg.tmp32.sroa.1.0.copyload
  %spec.select32.i = sub nsw i32 %21, %sub57.i
  %mul60.i = mul i32 %spec.select32.i, %agg.tmp32.sroa.446.0.copyload
  %cmp36.not.i = icmp slt i32 %22, %agg.tmp32.sroa.2.0.copyload
  %sub38.i = select i1 %cmp36.not.i, i32 0, i32 %agg.tmp32.sroa.2.0.copyload
  %spec.select.i = sub nsw i32 %22, %sub38.i
  %28 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @llvm.amdgcn.kernel._Z24pmeSplineAndSpreadKernelILi4ELb1ELb1ELb1ELb1ELi1ELb0EL14ThreadsPerAtom0ELi32EEv18PmeGpuKernelParams.lds, i32 %add4.i.i
  %arrayidx42.i = getelementptr inbounds nuw i8, ptr addrspace(3) %28, i32 32
  %29 = load float, ptr addrspace(3) %arrayidx42.i, align 4, !tbaa !20, !noalias !39
  %mul43.i = fmul fast float %27, %13
  %mul44.i45 = fmul fast float %mul43.i, %29
  %reass.add.i = add i32 %mul60.i, %spec.select.i
  %reass.mul.i = mul i32 %reass.add.i, %agg.tmp32.sroa.5.0.copyload
  %add61.i = add i32 %add46.i, %reass.mul.i
  %30 = load float, ptr addrspace(3) %28, align 4, !tbaa !20, !noalias !39
  %idx.ext.i = sext i32 %add61.i to i64
  %add.ptr.i = getelementptr inbounds [4 x i8], ptr addrspace(1) %20, i64 %idx.ext.i
  %mul65.i = fmul fast float %mul44.i45, %30
  %31 = atomicrmw fadd ptr addrspace(1) %add.ptr.i, float %mul65.i syncscope("agent") monotonic, align 4, !noalias !40, !noalias.addrspace !41, !amdgpu.no.fine.grained.memory !16, !amdgpu.no.remote.memory !16, !amdgpu.ignore.denormal.mode !16
  %add51.1.i = add nsw i32 %21, 1
  %cmp52.not.1.i = icmp slt i32 %add51.1.i, %agg.tmp32.sroa.1.0.copyload
  %sub57.1.i = select i1 %cmp52.not.1.i, i32 0, i32 %agg.tmp32.sroa.1.0.copyload
  %spec.select32.1.i = sub nsw i32 %add51.1.i, %sub57.1.i
  %mul60.1.i = mul i32 %spec.select32.1.i, %agg.tmp32.sroa.446.0.copyload
  %reass.add.1.i = add i32 %mul60.1.i, %spec.select.i
  %reass.mul.1.i = mul i32 %reass.add.1.i, %agg.tmp32.sroa.5.0.copyload
  %add61.1.i = add i32 %reass.mul.1.i, %add46.i
  %arrayidx64.1.i = getelementptr inbounds nuw i8, ptr addrspace(3) %28, i32 96
  %32 = load float, ptr addrspace(3) %arrayidx64.1.i, align 4, !tbaa !20, !noalias !39
  %idx.ext.1.i = sext i32 %add61.1.i to i64
  %add.ptr.1.i = getelementptr inbounds [4 x i8], ptr addrspace(1) %20, i64 %idx.ext.1.i
  %mul65.1.i = fmul fast float %32, %mul44.i45
  %33 = atomicrmw fadd ptr addrspace(1) %add.ptr.1.i, float %mul65.1.i syncscope("agent") monotonic, align 4, !noalias !40, !noalias.addrspace !41, !amdgpu.no.fine.grained.memory !16, !amdgpu.no.remote.memory !16, !amdgpu.ignore.denormal.mode !16
  %add51.2.i = add nsw i32 %21, 2
  %cmp52.not.2.i = icmp slt i32 %add51.2.i, %agg.tmp32.sroa.1.0.copyload
  %sub57.2.i = select i1 %cmp52.not.2.i, i32 0, i32 %agg.tmp32.sroa.1.0.copyload
  %spec.select32.2.i = sub nsw i32 %add51.2.i, %sub57.2.i
  %mul60.2.i = mul i32 %spec.select32.2.i, %agg.tmp32.sroa.446.0.copyload
  %reass.add.2.i = add i32 %mul60.2.i, %spec.select.i
  %reass.mul.2.i = mul i32 %reass.add.2.i, %agg.tmp32.sroa.5.0.copyload
  %add61.2.i = add i32 %reass.mul.2.i, %add46.i
  %arrayidx64.2.i = getelementptr inbounds nuw i8, ptr addrspace(3) %28, i32 192
  %34 = load float, ptr addrspace(3) %arrayidx64.2.i, align 4, !tbaa !20, !noalias !39
  %idx.ext.2.i = sext i32 %add61.2.i to i64
  %add.ptr.2.i = getelementptr inbounds [4 x i8], ptr addrspace(1) %20, i64 %idx.ext.2.i
  %mul65.2.i = fmul fast float %34, %mul44.i45
  %35 = atomicrmw fadd ptr addrspace(1) %add.ptr.2.i, float %mul65.2.i syncscope("agent") monotonic, align 4, !noalias !40, !noalias.addrspace !41, !amdgpu.no.fine.grained.memory !16, !amdgpu.no.remote.memory !16, !amdgpu.ignore.denormal.mode !16
  %add51.3.i = add nsw i32 %21, 3
  %cmp52.not.3.i = icmp slt i32 %add51.3.i, %agg.tmp32.sroa.1.0.copyload
  %sub57.3.i = select i1 %cmp52.not.3.i, i32 0, i32 %agg.tmp32.sroa.1.0.copyload
  %spec.select32.3.i = sub nsw i32 %add51.3.i, %sub57.3.i
  %mul60.3.i = mul i32 %spec.select32.3.i, %agg.tmp32.sroa.446.0.copyload
  %reass.add.3.i = add i32 %mul60.3.i, %spec.select.i
  %reass.mul.3.i = mul i32 %reass.add.3.i, %agg.tmp32.sroa.5.0.copyload
  %add61.3.i = add i32 %reass.mul.3.i, %add46.i
  %arrayidx64.3.i = getelementptr inbounds nuw i8, ptr addrspace(3) %28, i32 288
  %36 = load float, ptr addrspace(3) %arrayidx64.3.i, align 4, !tbaa !20, !noalias !39
  %idx.ext.3.i = sext i32 %add61.3.i to i64
  %add.ptr.3.i = getelementptr inbounds [4 x i8], ptr addrspace(1) %20, i64 %idx.ext.3.i
  %mul65.3.i = fmul fast float %36, %mul44.i45
  %37 = atomicrmw fadd ptr addrspace(1) %add.ptr.3.i, float %mul65.3.i syncscope("agent") monotonic, align 4, !noalias !40, !noalias.addrspace !41, !amdgpu.no.fine.grained.memory !16, !amdgpu.no.remote.memory !16, !amdgpu.ignore.denormal.mode !16
  %add35.1.i = add nsw i32 %22, 1
  %cmp36.not.1.i = icmp slt i32 %add35.1.i, %agg.tmp32.sroa.2.0.copyload
  %sub38.1.i = select i1 %cmp36.not.1.i, i32 0, i32 %agg.tmp32.sroa.2.0.copyload
  %spec.select.1.i = sub nsw i32 %add35.1.i, %sub38.1.i
  %arrayidx42.1.i = getelementptr inbounds nuw i8, ptr addrspace(3) %28, i32 128
  %38 = load float, ptr addrspace(3) %arrayidx42.1.i, align 4, !tbaa !20, !noalias !39
  %mul44.1.i = fmul fast float %mul43.i, %38
  %reass.add.143.i = add i32 %spec.select.1.i, %mul60.i
  %reass.mul.144.i = mul i32 %reass.add.143.i, %agg.tmp32.sroa.5.0.copyload
  %add61.145.i = add i32 %reass.mul.144.i, %add46.i
  %idx.ext.148.i = sext i32 %add61.145.i to i64
  %add.ptr.149.i = getelementptr inbounds [4 x i8], ptr addrspace(1) %20, i64 %idx.ext.148.i
  %mul65.150.i = fmul fast float %mul44.1.i, %30
  %39 = atomicrmw fadd ptr addrspace(1) %add.ptr.149.i, float %mul65.150.i syncscope("agent") monotonic, align 4, !noalias !40, !noalias.addrspace !41, !amdgpu.no.fine.grained.memory !16, !amdgpu.no.remote.memory !16, !amdgpu.ignore.denormal.mode !16
  %reass.add.1.1.i = add i32 %mul60.1.i, %spec.select.1.i
  %reass.mul.1.1.i = mul i32 %reass.add.1.1.i, %agg.tmp32.sroa.5.0.copyload
  %add61.1.1.i = add i32 %reass.mul.1.1.i, %add46.i
  %idx.ext.1.1.i = sext i32 %add61.1.1.i to i64
  %add.ptr.1.1.i = getelementptr inbounds [4 x i8], ptr addrspace(1) %20, i64 %idx.ext.1.1.i
  %mul65.1.1.i = fmul fast float %mul44.1.i, %32
  %40 = atomicrmw fadd ptr addrspace(1) %add.ptr.1.1.i, float %mul65.1.1.i syncscope("agent") monotonic, align 4, !noalias !40, !noalias.addrspace !41, !amdgpu.no.fine.grained.memory !16, !amdgpu.no.remote.memory !16, !amdgpu.ignore.denormal.mode !16
  %reass.add.2.1.i = add i32 %mul60.2.i, %spec.select.1.i
  %reass.mul.2.1.i = mul i32 %reass.add.2.1.i, %agg.tmp32.sroa.5.0.copyload
  %add61.2.1.i = add i32 %reass.mul.2.1.i, %add46.i
  %idx.ext.2.1.i = sext i32 %add61.2.1.i to i64
  %add.ptr.2.1.i = getelementptr inbounds [4 x i8], ptr addrspace(1) %20, i64 %idx.ext.2.1.i
  %mul65.2.1.i = fmul fast float %mul44.1.i, %34
  %41 = atomicrmw fadd ptr addrspace(1) %add.ptr.2.1.i, float %mul65.2.1.i syncscope("agent") monotonic, align 4, !noalias !40, !noalias.addrspace !41, !amdgpu.no.fine.grained.memory !16, !amdgpu.no.remote.memory !16, !amdgpu.ignore.denormal.mode !16
  %reass.add.3.1.i = add i32 %mul60.3.i, %spec.select.1.i
  %reass.mul.3.1.i = mul i32 %reass.add.3.1.i, %agg.tmp32.sroa.5.0.copyload
  %add61.3.1.i = add i32 %reass.mul.3.1.i, %add46.i
  %idx.ext.3.1.i = sext i32 %add61.3.1.i to i64
  %add.ptr.3.1.i = getelementptr inbounds [4 x i8], ptr addrspace(1) %20, i64 %idx.ext.3.1.i
  %mul65.3.1.i = fmul fast float %mul44.1.i, %36
  %42 = atomicrmw fadd ptr addrspace(1) %add.ptr.3.1.i, float %mul65.3.1.i syncscope("agent") monotonic, align 4, !noalias !40, !noalias.addrspace !41, !amdgpu.no.fine.grained.memory !16, !amdgpu.no.remote.memory !16, !amdgpu.ignore.denormal.mode !16
  %add35.2.i = add nsw i32 %22, 2
  %cmp36.not.2.i = icmp slt i32 %add35.2.i, %agg.tmp32.sroa.2.0.copyload
  %sub38.2.i = select i1 %cmp36.not.2.i, i32 0, i32 %agg.tmp32.sroa.2.0.copyload
  %spec.select.2.i = sub nsw i32 %add35.2.i, %sub38.2.i
  %arrayidx42.2.i = getelementptr inbounds nuw i8, ptr addrspace(3) %28, i32 224
  %43 = load float, ptr addrspace(3) %arrayidx42.2.i, align 4, !tbaa !20, !noalias !39
  %mul44.2.i = fmul fast float %mul43.i, %43
  %reass.add.251.i = add i32 %spec.select.2.i, %mul60.i
  %reass.mul.252.i = mul i32 %reass.add.251.i, %agg.tmp32.sroa.5.0.copyload
  %add61.253.i = add i32 %reass.mul.252.i, %add46.i
  %idx.ext.256.i = sext i32 %add61.253.i to i64
  %add.ptr.257.i = getelementptr inbounds [4 x i8], ptr addrspace(1) %20, i64 %idx.ext.256.i
  %mul65.258.i = fmul fast float %mul44.2.i, %30
  %44 = atomicrmw fadd ptr addrspace(1) %add.ptr.257.i, float %mul65.258.i syncscope("agent") monotonic, align 4, !noalias !40, !noalias.addrspace !41, !amdgpu.no.fine.grained.memory !16, !amdgpu.no.remote.memory !16, !amdgpu.ignore.denormal.mode !16
  %reass.add.1.2.i = add i32 %mul60.1.i, %spec.select.2.i
  %reass.mul.1.2.i = mul i32 %reass.add.1.2.i, %agg.tmp32.sroa.5.0.copyload
  %add61.1.2.i = add i32 %reass.mul.1.2.i, %add46.i
  %idx.ext.1.2.i = sext i32 %add61.1.2.i to i64
  %add.ptr.1.2.i = getelementptr inbounds [4 x i8], ptr addrspace(1) %20, i64 %idx.ext.1.2.i
  %mul65.1.2.i = fmul fast float %mul44.2.i, %32
  %45 = atomicrmw fadd ptr addrspace(1) %add.ptr.1.2.i, float %mul65.1.2.i syncscope("agent") monotonic, align 4, !noalias !40, !noalias.addrspace !41, !amdgpu.no.fine.grained.memory !16, !amdgpu.no.remote.memory !16, !amdgpu.ignore.denormal.mode !16
  %reass.add.2.2.i = add i32 %mul60.2.i, %spec.select.2.i
  %reass.mul.2.2.i = mul i32 %reass.add.2.2.i, %agg.tmp32.sroa.5.0.copyload
  %add61.2.2.i = add i32 %reass.mul.2.2.i, %add46.i
  %idx.ext.2.2.i = sext i32 %add61.2.2.i to i64
  %add.ptr.2.2.i = getelementptr inbounds [4 x i8], ptr addrspace(1) %20, i64 %idx.ext.2.2.i
  %mul65.2.2.i = fmul fast float %mul44.2.i, %34
  %46 = atomicrmw fadd ptr addrspace(1) %add.ptr.2.2.i, float %mul65.2.2.i syncscope("agent") monotonic, align 4, !noalias !40, !noalias.addrspace !41, !amdgpu.no.fine.grained.memory !16, !amdgpu.no.remote.memory !16, !amdgpu.ignore.denormal.mode !16
  %reass.add.3.2.i = add i32 %mul60.3.i, %spec.select.2.i
  %reass.mul.3.2.i = mul i32 %reass.add.3.2.i, %agg.tmp32.sroa.5.0.copyload
  %add61.3.2.i = add i32 %reass.mul.3.2.i, %add46.i
  %idx.ext.3.2.i = sext i32 %add61.3.2.i to i64
  %add.ptr.3.2.i = getelementptr inbounds [4 x i8], ptr addrspace(1) %20, i64 %idx.ext.3.2.i
  %mul65.3.2.i = fmul fast float %mul44.2.i, %36
  %47 = atomicrmw fadd ptr addrspace(1) %add.ptr.3.2.i, float %mul65.3.2.i syncscope("agent") monotonic, align 4, !noalias !40, !noalias.addrspace !41, !amdgpu.no.fine.grained.memory !16, !amdgpu.no.remote.memory !16, !amdgpu.ignore.denormal.mode !16
  %add35.3.i = add nsw i32 %22, 3
  %cmp36.not.3.i = icmp slt i32 %add35.3.i, %agg.tmp32.sroa.2.0.copyload
  %sub38.3.i = select i1 %cmp36.not.3.i, i32 0, i32 %agg.tmp32.sroa.2.0.copyload
  %spec.select.3.i = sub nsw i32 %add35.3.i, %sub38.3.i
  %arrayidx42.3.i = getelementptr inbounds nuw i8, ptr addrspace(3) %28, i32 320
  %48 = load float, ptr addrspace(3) %arrayidx42.3.i, align 4, !tbaa !20, !noalias !39
  %mul44.3.i = fmul fast float %mul43.i, %48
  %reass.add.359.i = add i32 %spec.select.3.i, %mul60.i
  %reass.mul.360.i = mul i32 %reass.add.359.i, %agg.tmp32.sroa.5.0.copyload
  %add61.361.i = add i32 %reass.mul.360.i, %add46.i
  %idx.ext.364.i = sext i32 %add61.361.i to i64
  %add.ptr.365.i = getelementptr inbounds [4 x i8], ptr addrspace(1) %20, i64 %idx.ext.364.i
  %mul65.366.i = fmul fast float %mul44.3.i, %30
  %49 = atomicrmw fadd ptr addrspace(1) %add.ptr.365.i, float %mul65.366.i syncscope("agent") monotonic, align 4, !noalias !40, !noalias.addrspace !41, !amdgpu.no.fine.grained.memory !16, !amdgpu.no.remote.memory !16, !amdgpu.ignore.denormal.mode !16
  %reass.add.1.3.i = add i32 %mul60.1.i, %spec.select.3.i
  %reass.mul.1.3.i = mul i32 %reass.add.1.3.i, %agg.tmp32.sroa.5.0.copyload
  %add61.1.3.i = add i32 %reass.mul.1.3.i, %add46.i
  %idx.ext.1.3.i = sext i32 %add61.1.3.i to i64
  %add.ptr.1.3.i = getelementptr inbounds [4 x i8], ptr addrspace(1) %20, i64 %idx.ext.1.3.i
  %mul65.1.3.i = fmul fast float %mul44.3.i, %32
  %50 = atomicrmw fadd ptr addrspace(1) %add.ptr.1.3.i, float %mul65.1.3.i syncscope("agent") monotonic, align 4, !noalias !40, !noalias.addrspace !41, !amdgpu.no.fine.grained.memory !16, !amdgpu.no.remote.memory !16, !amdgpu.ignore.denormal.mode !16
  %reass.add.2.3.i = add i32 %mul60.2.i, %spec.select.3.i
  %reass.mul.2.3.i = mul i32 %reass.add.2.3.i, %agg.tmp32.sroa.5.0.copyload
  %add61.2.3.i = add i32 %reass.mul.2.3.i, %add46.i
  %idx.ext.2.3.i = sext i32 %add61.2.3.i to i64
  %add.ptr.2.3.i = getelementptr inbounds [4 x i8], ptr addrspace(1) %20, i64 %idx.ext.2.3.i
  %mul65.2.3.i = fmul fast float %mul44.3.i, %34
  %51 = atomicrmw fadd ptr addrspace(1) %add.ptr.2.3.i, float %mul65.2.3.i syncscope("agent") monotonic, align 4, !noalias !40, !noalias.addrspace !41, !amdgpu.no.fine.grained.memory !16, !amdgpu.no.remote.memory !16, !amdgpu.ignore.denormal.mode !16
  %reass.add.3.3.i = add i32 %mul60.3.i, %spec.select.3.i
  %reass.mul.3.3.i = mul i32 %reass.add.3.3.i, %agg.tmp32.sroa.5.0.copyload
  %add61.3.3.i = add i32 %reass.mul.3.3.i, %add46.i
  %idx.ext.3.3.i = sext i32 %add61.3.3.i to i64
  %add.ptr.3.3.i = getelementptr inbounds [4 x i8], ptr addrspace(1) %20, i64 %idx.ext.3.3.i
  %mul65.3.3.i = fmul fast float %mul44.3.i, %36
  %52 = atomicrmw fadd ptr addrspace(1) %add.ptr.3.3.i, float %mul65.3.3.i syncscope("agent") monotonic, align 4, !noalias !40, !noalias.addrspace !41, !amdgpu.no.fine.grained.memory !16, !amdgpu.no.remote.memory !16, !amdgpu.ignore.denormal.mode !16
  br label %cleanup

cleanup:                                          ; preds = %if.then31, %land.lhs.true, %_ZL17calculate_splinesILi4ELi32ELi8ELb0ELb0ELi1ELi32EEv18PmeGpuKernelParamsi15HIP_vector_typeIfLj3EEfPfS3_Pi.exit, %entry
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #2 = { convergent nocallback nofree nounwind willreturn }
attributes #3 = { convergent mustprogress nofree norecurse nounwind willreturn "amdgpu-assume-32bit-global-offset"="true" "amdgpu-assume-non-negative-fract"="true" "amdgpu-flat-work-group-size"="128,128" "amdgpu-lds-size"="1920" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-wwm" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1201" "uniform-work-group-size" }

!llvm.ident = !{!1, !2}
!llvm.errno.tbaa = !{!3, !7}
!opencl.ocl.version = !{!11}
!llvm.module.flags = !{!12, !13, !14, !15}

!0 = !{i32 0, i32 1}
!1 = !{!"clang version 23.0.0git (https://github.com/gandhi56/llvm-project c917fe3b79c9559da0f53c6d9ff15663aa9c5c42)"}
!2 = !{!"AMD clang version 22.0.0git (https://github.com/ROCm/llvm-project.git c849bc16b0e49951d313756f20b73c2b28d321d7+PATCHED:9a6ac45c97a1e511db838c5b46257324d2de1780)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!8, !8, i64 0}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{i32 2, i32 0}
!12 = !{i32 1, !"amdhsa_code_object_version", i32 600}
!13 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!14 = !{i32 8, !"PIC Level", i32 2}
!15 = !{i32 1, !"wchar_size", i32 4}
!16 = !{}
!17 = !{!18, !18, i64 0}
!18 = !{!"short", !9, i64 0}
!19 = !{i16 1, i16 1025}
!20 = !{!21, !21, i64 0}
!21 = !{!"float", !5, i64 0}
!22 = !{!5, !5, i64 0}
!23 = !{!24, !24, i64 0}
!24 = !{!"p1 float", !25, i64 0}
!25 = !{!"any pointer", !5, i64 0}
!26 = !{!27, !27, i64 0}
!27 = !{!"p1 int", !25, i64 0}
!28 = !{!29}
!29 = distinct !{!29, !30}
!30 = distinct !{!30}
!31 = !{!32}
!32 = distinct !{!32, !30}
!33 = !{!34}
!34 = distinct !{!34, !35, !"_Z14spread_chargesILi4ELb1ELb1ELi0EL14ThreadsPerAtom0ELi32EEv18PmeGpuKernelParamsPKfPKiS3_: %sm_gridlineIndices"}
!35 = distinct !{!35, !"_Z14spread_chargesILi4ELb1ELb1ELi0EL14ThreadsPerAtom0ELi32EEv18PmeGpuKernelParamsPKfPKiS3_"}
!36 = !{!37}
!37 = distinct !{!37, !35, !"_Z14spread_chargesILi4ELb1ELb1ELi0EL14ThreadsPerAtom0ELi32EEv18PmeGpuKernelParamsPKfPKiS3_: %sm_theta"}
!38 = !{!37, !32}
!39 = !{!34, !29}
!40 = !{!34, !37}
!41 = !{i32 5, i32 6}
