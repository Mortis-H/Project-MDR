
; __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa--gfx950
; ModuleID = 'multi_kernel.hip'
source_filename = "multi_kernel.hip"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@__hip_cuid_1c2b7cccb52ddebe = addrspace(1) global i8 0
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_1c2b7cccb52ddebe to ptr)], section "llvm.metadata"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define protected amdgpu_kernel void @vectorAdd(ptr addrspace(1) nocapture noundef readonly %0, ptr addrspace(1) nocapture noundef readonly %1, ptr addrspace(1) nocapture noundef writeonly %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %6 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %7 = getelementptr inbounds nuw i8, ptr addrspace(4) %6, i64 12
  %8 = load i16, ptr addrspace(4) %7, align 4, !tbaa !6
  %9 = zext i16 %8 to i32
  %10 = mul i32 %5, %9
  %11 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %12 = add i32 %10, %11
  %13 = icmp slt i32 %12, %3
  br i1 %13, label %14, label %22

14:                                               ; preds = %4
  %15 = sext i32 %12 to i64
  %16 = getelementptr inbounds float, ptr addrspace(1) %2, i64 %15
  %17 = getelementptr inbounds float, ptr addrspace(1) %1, i64 %15
  %18 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %15
  %19 = load float, ptr addrspace(1) %18, align 4, !tbaa !10
  %20 = load float, ptr addrspace(1) %17, align 4, !tbaa !10
  %21 = fadd contract float %19, %20
  store float %21, ptr addrspace(1) %16, align 4, !tbaa !10
  br label %22

22:                                               ; preds = %14, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define protected amdgpu_kernel void @vectorMul(ptr addrspace(1) nocapture noundef readonly %0, ptr addrspace(1) nocapture noundef readonly %1, ptr addrspace(1) nocapture noundef writeonly %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %6 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %7 = getelementptr inbounds nuw i8, ptr addrspace(4) %6, i64 12
  %8 = load i16, ptr addrspace(4) %7, align 4, !tbaa !6
  %9 = zext i16 %8 to i32
  %10 = mul i32 %5, %9
  %11 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %12 = add i32 %10, %11
  %13 = icmp slt i32 %12, %3
  br i1 %13, label %14, label %22

14:                                               ; preds = %4
  %15 = sext i32 %12 to i64
  %16 = getelementptr inbounds float, ptr addrspace(1) %2, i64 %15
  %17 = getelementptr inbounds float, ptr addrspace(1) %1, i64 %15
  %18 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %15
  %19 = load float, ptr addrspace(1) %18, align 4, !tbaa !10
  %20 = load float, ptr addrspace(1) %17, align 4, !tbaa !10
  %21 = fmul contract float %19, %20
  store float %21, ptr addrspace(1) %16, align 4, !tbaa !10
  br label %22

22:                                               ; preds = %14, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define protected amdgpu_kernel void @vectorSub(ptr addrspace(1) nocapture noundef readonly %0, ptr addrspace(1) nocapture noundef readonly %1, ptr addrspace(1) nocapture noundef writeonly %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %6 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %7 = getelementptr inbounds nuw i8, ptr addrspace(4) %6, i64 12
  %8 = load i16, ptr addrspace(4) %7, align 4, !tbaa !6
  %9 = zext i16 %8 to i32
  %10 = mul i32 %5, %9
  %11 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %12 = add i32 %10, %11
  %13 = icmp slt i32 %12, %3
  br i1 %13, label %14, label %22

14:                                               ; preds = %4
  %15 = sext i32 %12 to i64
  %16 = getelementptr inbounds float, ptr addrspace(1) %2, i64 %15
  %17 = getelementptr inbounds float, ptr addrspace(1) %1, i64 %15
  %18 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %15
  %19 = load float, ptr addrspace(1) %18, align 4, !tbaa !10
  %20 = load float, ptr addrspace(1) %17, align 4, !tbaa !10
  %21 = fsub contract float %19, %20
  store float %21, ptr addrspace(1) %16, align 4, !tbaa !10
  br label %22

22:                                               ; preds = %14, %4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write)
define protected amdgpu_kernel void @scalarOps(ptr addrspace(1) nocapture noundef writeonly %0, i32 noundef %1) local_unnamed_addr #1 {
  %3 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %4 = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %5 = getelementptr inbounds nuw i8, ptr addrspace(4) %4, i64 12
  %6 = load i16, ptr addrspace(4) %5, align 4, !tbaa !6
  %7 = zext i16 %6 to i32
  %8 = mul i32 %3, %7
  %9 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %10 = add i32 %8, %9
  %11 = icmp slt i32 %10, %1
  br i1 %11, label %12, label %23

12:                                               ; preds = %2
  %13 = add nsw i32 %10, 1
  %14 = xor i32 %13, %10
  %15 = ashr i32 %10, 2
  %16 = shl i32 %13, 1
  %17 = add nsw i32 %13, %10
  %18 = sub i32 %17, %14
  %19 = mul nsw i32 %16, %15
  %20 = add nsw i32 %18, %19
  %21 = sext i32 %10 to i64
  %22 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %21
  store i32 %20, ptr addrspace(1) %22, align 4, !tbaa !14
  br label %23

23:                                               ; preds = %12, %2
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef align 4 ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "amdgpu-flat-work-group-size"="1,1024" "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx950" "target-features"="+16-bit-insts,+ashr-pk-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-global-pk-add-bf16-inst,+bf8-cvt-scale-insts,+bitop3-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot12-insts,+dot13-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+f16bf16-to-fp6bf6-cvt-scale-insts,+f32-to-f16bf16-cvt-sr-insts,+fp4-cvt-scale-insts,+fp6bf6-cvt-scale-insts,+fp8-conversion-insts,+fp8-cvt-scale-insts,+fp8-insts,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+gfx940-insts,+gfx950-insts,+mai-insts,+permlane16-swap,+permlane32-swap,+prng-inst,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" "uniform-work-group-size"="true" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) "amdgpu-flat-work-group-size"="1,1024" "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx950" "target-features"="+16-bit-insts,+ashr-pk-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-global-pk-add-bf16-inst,+bf8-cvt-scale-insts,+bitop3-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot12-insts,+dot13-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+f16bf16-to-fp6bf6-cvt-scale-insts,+f32-to-f16bf16-cvt-sr-insts,+fp4-cvt-scale-insts,+fp6bf6-cvt-scale-insts,+fp8-conversion-insts,+fp8-cvt-scale-insts,+fp8-insts,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+gfx940-insts,+gfx950-insts,+mai-insts,+permlane16-swap,+permlane32-swap,+prng-inst,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" "uniform-work-group-size"="true" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!opencl.ocl.version = !{!4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"amdhsa_code_object_version", i32 600}
!1 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{i32 8, !"PIC Level", i32 2}
!4 = !{i32 2, i32 0}
!5 = !{!"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.0.1 25314 f4087f6b428f0e6f575ebac8a8a724dab123d06e)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"short", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"float", !12, i64 0}
!12 = !{!"omnipotent char", !13, i64 0}
!13 = !{!"Simple C++ TBAA"}
!14 = !{!15, !15, i64 0}
!15 = !{!"int", !12, i64 0}

; __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn-amd-amdhsa--gfx950

; __CLANG_OFFLOAD_BUNDLE____START__ host-x86_64-unknown-linux-gnu-
; ModuleID = 'multi_kernel.hip'
source_filename = "multi_kernel.hip"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.dim3 = type { i32, i32, i32 }

@vectorAdd = dso_local constant ptr @__device_stub__vectorAdd, align 8
@vectorMul = dso_local constant ptr @__device_stub__vectorMul, align 8
@vectorSub = dso_local constant ptr @__device_stub__vectorSub, align 8
@scalarOps = dso_local constant ptr @__device_stub__scalarOps, align 8
@0 = private unnamed_addr constant [10 x i8] c"vectorAdd\00", align 1
@1 = private unnamed_addr constant [10 x i8] c"vectorMul\00", align 1
@2 = private unnamed_addr constant [10 x i8] c"vectorSub\00", align 1
@3 = private unnamed_addr constant [10 x i8] c"scalarOps\00", align 1
@__hip_fatbin_1c2b7cccb52ddebe = external constant i8, section ".hip_fatbin"
@__hip_fatbin_wrapper = internal constant { i32, i32, ptr, ptr } { i32 1212764230, i32 1, ptr @__hip_fatbin_1c2b7cccb52ddebe, ptr null }, section ".hipFatBinSegment", align 8
@__hip_gpubin_handle_1c2b7cccb52ddebe = internal unnamed_addr global ptr null, align 8
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @__hip_module_ctor, ptr null }]
@__hip_cuid_1c2b7cccb52ddebe = global i8 0
@llvm.compiler.used = appending global [1 x ptr] [ptr @__hip_cuid_1c2b7cccb52ddebe], section "llvm.metadata"

; Function Attrs: mustprogress norecurse uwtable
define dso_local void @__device_stub__vectorAdd(ptr noundef %0, ptr noundef %1, ptr noundef %2, i32 noundef %3) #0 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca i32, align 4
  %9 = alloca %struct.dim3, align 8
  %10 = alloca %struct.dim3, align 8
  %11 = alloca i64, align 8
  %12 = alloca ptr, align 8
  store ptr %0, ptr %5, align 8, !tbaa !3
  store ptr %1, ptr %6, align 8, !tbaa !3
  store ptr %2, ptr %7, align 8, !tbaa !3
  store i32 %3, ptr %8, align 4, !tbaa !8
  %13 = alloca [4 x ptr], align 16
  store ptr %5, ptr %13, align 16
  %14 = getelementptr inbounds nuw i8, ptr %13, i64 8
  store ptr %6, ptr %14, align 8
  %15 = getelementptr inbounds nuw i8, ptr %13, i64 16
  store ptr %7, ptr %15, align 16
  %16 = getelementptr inbounds nuw i8, ptr %13, i64 24
  store ptr %8, ptr %16, align 8
  %17 = call i32 @__hipPopCallConfiguration(ptr nonnull %9, ptr nonnull %10, ptr nonnull %11, ptr nonnull %12)
  %18 = load i64, ptr %11, align 8
  %19 = load ptr, ptr %12, align 8
  %20 = load i64, ptr %9, align 8
  %21 = getelementptr inbounds nuw i8, ptr %9, i64 8
  %22 = load i32, ptr %21, align 8
  %23 = load i64, ptr %10, align 8
  %24 = getelementptr inbounds nuw i8, ptr %10, i64 8
  %25 = load i32, ptr %24, align 8
  %26 = call noundef i32 @hipLaunchKernel(ptr noundef nonnull @vectorAdd, i64 %20, i32 %22, i64 %23, i32 %25, ptr noundef nonnull %13, i64 noundef %18, ptr noundef %19)
  ret void
}

declare dso_local i32 @__hipPopCallConfiguration(ptr, ptr, ptr, ptr) local_unnamed_addr

declare dso_local i32 @hipLaunchKernel(ptr, i64, i32, i64, i32, ptr, i64, ptr) local_unnamed_addr

; Function Attrs: mustprogress norecurse uwtable
define dso_local void @__device_stub__vectorMul(ptr noundef %0, ptr noundef %1, ptr noundef %2, i32 noundef %3) #0 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca i32, align 4
  %9 = alloca %struct.dim3, align 8
  %10 = alloca %struct.dim3, align 8
  %11 = alloca i64, align 8
  %12 = alloca ptr, align 8
  store ptr %0, ptr %5, align 8, !tbaa !3
  store ptr %1, ptr %6, align 8, !tbaa !3
  store ptr %2, ptr %7, align 8, !tbaa !3
  store i32 %3, ptr %8, align 4, !tbaa !8
  %13 = alloca [4 x ptr], align 16
  store ptr %5, ptr %13, align 16
  %14 = getelementptr inbounds nuw i8, ptr %13, i64 8
  store ptr %6, ptr %14, align 8
  %15 = getelementptr inbounds nuw i8, ptr %13, i64 16
  store ptr %7, ptr %15, align 16
  %16 = getelementptr inbounds nuw i8, ptr %13, i64 24
  store ptr %8, ptr %16, align 8
  %17 = call i32 @__hipPopCallConfiguration(ptr nonnull %9, ptr nonnull %10, ptr nonnull %11, ptr nonnull %12)
  %18 = load i64, ptr %11, align 8
  %19 = load ptr, ptr %12, align 8
  %20 = load i64, ptr %9, align 8
  %21 = getelementptr inbounds nuw i8, ptr %9, i64 8
  %22 = load i32, ptr %21, align 8
  %23 = load i64, ptr %10, align 8
  %24 = getelementptr inbounds nuw i8, ptr %10, i64 8
  %25 = load i32, ptr %24, align 8
  %26 = call noundef i32 @hipLaunchKernel(ptr noundef nonnull @vectorMul, i64 %20, i32 %22, i64 %23, i32 %25, ptr noundef nonnull %13, i64 noundef %18, ptr noundef %19)
  ret void
}

; Function Attrs: mustprogress norecurse uwtable
define dso_local void @__device_stub__vectorSub(ptr noundef %0, ptr noundef %1, ptr noundef %2, i32 noundef %3) #0 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca i32, align 4
  %9 = alloca %struct.dim3, align 8
  %10 = alloca %struct.dim3, align 8
  %11 = alloca i64, align 8
  %12 = alloca ptr, align 8
  store ptr %0, ptr %5, align 8, !tbaa !3
  store ptr %1, ptr %6, align 8, !tbaa !3
  store ptr %2, ptr %7, align 8, !tbaa !3
  store i32 %3, ptr %8, align 4, !tbaa !8
  %13 = alloca [4 x ptr], align 16
  store ptr %5, ptr %13, align 16
  %14 = getelementptr inbounds nuw i8, ptr %13, i64 8
  store ptr %6, ptr %14, align 8
  %15 = getelementptr inbounds nuw i8, ptr %13, i64 16
  store ptr %7, ptr %15, align 16
  %16 = getelementptr inbounds nuw i8, ptr %13, i64 24
  store ptr %8, ptr %16, align 8
  %17 = call i32 @__hipPopCallConfiguration(ptr nonnull %9, ptr nonnull %10, ptr nonnull %11, ptr nonnull %12)
  %18 = load i64, ptr %11, align 8
  %19 = load ptr, ptr %12, align 8
  %20 = load i64, ptr %9, align 8
  %21 = getelementptr inbounds nuw i8, ptr %9, i64 8
  %22 = load i32, ptr %21, align 8
  %23 = load i64, ptr %10, align 8
  %24 = getelementptr inbounds nuw i8, ptr %10, i64 8
  %25 = load i32, ptr %24, align 8
  %26 = call noundef i32 @hipLaunchKernel(ptr noundef nonnull @vectorSub, i64 %20, i32 %22, i64 %23, i32 %25, ptr noundef nonnull %13, i64 noundef %18, ptr noundef %19)
  ret void
}

; Function Attrs: mustprogress norecurse uwtable
define dso_local void @__device_stub__scalarOps(ptr noundef %0, i32 noundef %1) #0 {
  %3 = alloca ptr, align 8
  %4 = alloca i32, align 4
  %5 = alloca %struct.dim3, align 8
  %6 = alloca %struct.dim3, align 8
  %7 = alloca i64, align 8
  %8 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8, !tbaa !10
  store i32 %1, ptr %4, align 4, !tbaa !8
  %9 = alloca [2 x ptr], align 16
  store ptr %3, ptr %9, align 16
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 8
  store ptr %4, ptr %10, align 8
  %11 = call i32 @__hipPopCallConfiguration(ptr nonnull %5, ptr nonnull %6, ptr nonnull %7, ptr nonnull %8)
  %12 = load i64, ptr %7, align 8
  %13 = load ptr, ptr %8, align 8
  %14 = load i64, ptr %5, align 8
  %15 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %16 = load i32, ptr %15, align 8
  %17 = load i64, ptr %6, align 8
  %18 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %19 = load i32, ptr %18, align 8
  %20 = call noundef i32 @hipLaunchKernel(ptr noundef nonnull @scalarOps, i64 %14, i32 %16, i64 %17, i32 %19, ptr noundef nonnull %9, i64 noundef %12, ptr noundef %13)
  ret void
}

declare dso_local i32 @__hipRegisterFunction(ptr, ptr, ptr, ptr, i32, ptr, ptr, ptr, ptr, ptr) local_unnamed_addr

declare dso_local ptr @__hipRegisterFatBinary(ptr) local_unnamed_addr

define internal void @__hip_module_ctor() {
  %1 = load ptr, ptr @__hip_gpubin_handle_1c2b7cccb52ddebe, align 8
  %2 = icmp eq ptr %1, null
  br i1 %2, label %3, label %5

3:                                                ; preds = %0
  %4 = tail call ptr @__hipRegisterFatBinary(ptr nonnull @__hip_fatbin_wrapper)
  store ptr %4, ptr @__hip_gpubin_handle_1c2b7cccb52ddebe, align 8
  br label %5

5:                                                ; preds = %3, %0
  %6 = phi ptr [ %4, %3 ], [ %1, %0 ]
  %7 = tail call i32 @__hipRegisterFunction(ptr %6, ptr nonnull @vectorAdd, ptr nonnull @0, ptr nonnull @0, i32 -1, ptr null, ptr null, ptr null, ptr null, ptr null)
  %8 = tail call i32 @__hipRegisterFunction(ptr %6, ptr nonnull @vectorMul, ptr nonnull @1, ptr nonnull @1, i32 -1, ptr null, ptr null, ptr null, ptr null, ptr null)
  %9 = tail call i32 @__hipRegisterFunction(ptr %6, ptr nonnull @vectorSub, ptr nonnull @2, ptr nonnull @2, i32 -1, ptr null, ptr null, ptr null, ptr null, ptr null)
  %10 = tail call i32 @__hipRegisterFunction(ptr %6, ptr nonnull @scalarOps, ptr nonnull @3, ptr nonnull @3, i32 -1, ptr null, ptr null, ptr null, ptr null, ptr null)
  %11 = tail call i32 @atexit(ptr nonnull @__hip_module_dtor)
  ret void
}

declare dso_local void @__hipUnregisterFatBinary(ptr) local_unnamed_addr

define internal void @__hip_module_dtor() {
  %1 = load ptr, ptr @__hip_gpubin_handle_1c2b7cccb52ddebe, align 8
  %2 = icmp eq ptr %1, null
  br i1 %2, label %4, label %3

3:                                                ; preds = %0
  tail call void @__hipUnregisterFatBinary(ptr nonnull %1)
  store ptr null, ptr @__hip_gpubin_handle_1c2b7cccb52ddebe, align 8
  br label %4

4:                                                ; preds = %3, %0
  ret void
}

; Function Attrs: nofree
declare dso_local i32 @atexit(ptr) local_unnamed_addr #1

attributes #0 = { mustprogress norecurse uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "uniform-work-group-size"="true" }
attributes #1 = { nofree }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 2}
!2 = !{!"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.0.1 25314 f4087f6b428f0e6f575ebac8a8a724dab123d06e)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"p1 float", !5, i64 0}
!5 = !{!"any pointer", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C++ TBAA"}
!8 = !{!9, !9, i64 0}
!9 = !{!"int", !6, i64 0}
!10 = !{!11, !11, i64 0}
!11 = !{!"p1 int", !5, i64 0}

; __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu-
