; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --version 5
; RUN: opt -passes=constraint-elimination -S %s | FileCheck %s

define i1 @test_icmp_ult_zext_icmp_trunc_nuw(i16 %x, i32 %y) {
; CHECK-LABEL: define i1 @test_icmp_ult_zext_icmp_trunc_nuw(
; CHECK-SAME: i16 [[X:%.*]], i32 [[Y:%.*]]) {
; CHECK-NEXT:    [[EXT:%.*]] = zext i16 [[X]] to i32
; CHECK-NEXT:    [[COND:%.*]] = icmp ult i32 [[Y]], [[EXT]]
; CHECK-NEXT:    br i1 [[COND]], label %[[IF_THEN:.*]], label %[[IF_ELSE:.*]]
; CHECK:       [[IF_THEN]]:
; CHECK-NEXT:    [[CONV:%.*]] = trunc nuw i32 [[Y]] to i16
; CHECK-NEXT:    ret i1 false
; CHECK:       [[IF_ELSE]]:
; CHECK-NEXT:    ret i1 false
;
  %ext = zext i16 %x to i32
  %cond = icmp ult i32 %y, %ext
  br i1 %cond, label %if.then, label %if.else

if.then:
  %conv = trunc nuw i32 %y to i16
  %cmp = icmp eq i16 %x, %conv
  ret i1 %cmp

if.else:
  ret i1 false
}

define i1 @test_icmp_slt_sext_icmp_trunc_nsw(i16 %x, i32 %y) {
; CHECK-LABEL: define i1 @test_icmp_slt_sext_icmp_trunc_nsw(
; CHECK-SAME: i16 [[X:%.*]], i32 [[Y:%.*]]) {
; CHECK-NEXT:    [[EXT:%.*]] = sext i16 [[X]] to i32
; CHECK-NEXT:    [[COND:%.*]] = icmp slt i32 [[Y]], [[EXT]]
; CHECK-NEXT:    br i1 [[COND]], label %[[IF_THEN:.*]], label %[[IF_ELSE:.*]]
; CHECK:       [[IF_THEN]]:
; CHECK-NEXT:    [[CONV:%.*]] = trunc nsw i32 [[Y]] to i16
; CHECK-NEXT:    ret i1 false
; CHECK:       [[IF_ELSE]]:
; CHECK-NEXT:    ret i1 false
;
  %ext = sext i16 %x to i32
  %cond = icmp slt i32 %y, %ext
  br i1 %cond, label %if.then, label %if.else

if.then:
  %conv = trunc nsw i32 %y to i16
  %cmp = icmp slt i16 %x, %conv
  ret i1 %cmp

if.else:
  ret i1 false
}

define i1 @test_icmp_ult_trunc_nsw_nneg_icmp_trunc_nuw(i64 %x, i32 %y) {
; CHECK-LABEL: define i1 @test_icmp_ult_trunc_nsw_nneg_icmp_trunc_nuw(
; CHECK-SAME: i64 [[X:%.*]], i32 [[Y:%.*]]) {
; CHECK-NEXT:    [[EXT:%.*]] = trunc nsw i64 [[X]] to i32
; CHECK-NEXT:    [[NNEG:%.*]] = icmp sgt i64 [[X]], -1
; CHECK-NEXT:    [[COND:%.*]] = icmp ult i32 [[Y]], [[EXT]]
; CHECK-NEXT:    [[AND:%.*]] = and i1 [[NNEG]], [[COND]]
; CHECK-NEXT:    br i1 [[AND]], label %[[IF_THEN:.*]], label %[[IF_ELSE:.*]]
; CHECK:       [[IF_THEN]]:
; CHECK-NEXT:    [[CONV:%.*]] = zext i32 [[Y]] to i64
; CHECK-NEXT:    ret i1 false
; CHECK:       [[IF_ELSE]]:
; CHECK-NEXT:    ret i1 false
;
  %ext = trunc nsw i64 %x to i32
  %nneg = icmp sgt i64 %x, -1
  %cond = icmp ult i32 %y, %ext
  %and = and i1 %nneg, %cond
  br i1 %and, label %if.then, label %if.else

if.then:
  %conv = zext i32 %y to i64
  %cmp = icmp eq i64 %x, %conv
  ret i1 %cmp

if.else:
  ret i1 false
}

define i1 @test2(i32 %n) {
; CHECK-LABEL: define i1 @test2(
; CHECK-SAME: i32 [[N:%.*]]) {
; CHECK-NEXT:    [[COND:%.*]] = icmp sgt i32 [[N]], 0
; CHECK-NEXT:    br i1 [[COND]], label %[[IF_THEN:.*]], label %[[IF_ELSE:.*]]
; CHECK:       [[IF_THEN]]:
; CHECK-NEXT:    [[EXT:%.*]] = zext nneg i32 [[N]] to i64
; CHECK-NEXT:    [[END:%.*]] = add nsw i64 [[EXT]], -1
; CHECK-NEXT:    br label %[[FOR_BODY:.*]]
; CHECK:       [[FOR_BODY]]:
; CHECK-NEXT:    [[INDVAR:%.*]] = phi i64 [ 0, %[[IF_THEN]] ], [ [[INDVAR_NEXT:%.*]], %[[FOR_NEXT:.*]] ]
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i64 [[INDVAR]], [[END]]
; CHECK-NEXT:    br i1 [[CMP]], label %[[IF_ELSE]], label %[[FOR_NEXT]]
; CHECK:       [[FOR_NEXT]]:
; CHECK-NEXT:    [[INDVAR_NEXT]] = add nuw nsw i64 [[INDVAR]], 1
; CHECK-NEXT:    [[COND2:%.*]] = call i1 @cond()
; CHECK-NEXT:    br i1 [[COND2]], label %[[FOR_BODY]], label %[[FOR_END:.*]]
; CHECK:       [[FOR_END]]:
; CHECK-NEXT:    [[TRUNC:%.*]] = trunc nsw i64 [[INDVAR_NEXT]] to i32
; CHECK-NEXT:    ret i1 true
; CHECK:       [[IF_ELSE]]:
; CHECK-NEXT:    ret i1 false
;
  %cond = icmp sgt i32 %n, 0
  br i1 %cond, label %if.then, label %if.else

if.then:
  %ext = zext nneg i32 %n to i64
  %end = add nsw i64 %ext, -1
  br label %for.body

for.body:
  %indvar = phi i64 [ 0, %if.then ], [ %indvar.next, %for.next ]
  %cmp = icmp eq i64 %indvar, %end
  br i1 %cmp, label %if.else, label %for.next

for.next:
  %indvar.next = add nuw nsw i64 %indvar, 1
  %cond2 = call i1 @cond()
  br i1 %cond2, label %for.body, label %for.end

for.end:
  %trunc = trunc nsw i64 %indvar.next to i32
  %res = icmp sgt i32 %n, %trunc
  ret i1 %res

if.else:
  ret i1 false
}

define i1 @test_icmp_ult_zext_icmp_trunc(i16 %x, i32 %y) {
; CHECK-LABEL: define i1 @test_icmp_ult_zext_icmp_trunc(
; CHECK-SAME: i16 [[X:%.*]], i32 [[Y:%.*]]) {
; CHECK-NEXT:    [[EXT:%.*]] = zext i16 [[X]] to i32
; CHECK-NEXT:    [[COND:%.*]] = icmp ult i32 [[Y]], [[EXT]]
; CHECK-NEXT:    br i1 [[COND]], label %[[IF_THEN:.*]], label %[[IF_ELSE:.*]]
; CHECK:       [[IF_THEN]]:
; CHECK-NEXT:    [[CONV:%.*]] = trunc i32 [[Y]] to i16
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i16 [[X]], [[CONV]]
; CHECK-NEXT:    ret i1 [[CMP]]
; CHECK:       [[IF_ELSE]]:
; CHECK-NEXT:    ret i1 false
;
  %ext = zext i16 %x to i32
  %cond = icmp ult i32 %y, %ext
  br i1 %cond, label %if.then, label %if.else

if.then:
  %conv = trunc i32 %y to i16
  %cmp = icmp eq i16 %x, %conv
  ret i1 %cmp

if.else:
  ret i1 false
}

define i1 @test_icmp_ult_zext_icmp_trunc_nuw_i128(i16 %x, i128 %y) {
; CHECK-LABEL: define i1 @test_icmp_ult_zext_icmp_trunc_nuw_i128(
; CHECK-SAME: i16 [[X:%.*]], i128 [[Y:%.*]]) {
; CHECK-NEXT:    [[EXT:%.*]] = zext i16 [[X]] to i128
; CHECK-NEXT:    [[COND:%.*]] = icmp ult i128 [[Y]], [[EXT]]
; CHECK-NEXT:    br i1 [[COND]], label %[[IF_THEN:.*]], label %[[IF_ELSE:.*]]
; CHECK:       [[IF_THEN]]:
; CHECK-NEXT:    [[CONV:%.*]] = trunc nuw i128 [[Y]] to i16
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i16 [[X]], [[CONV]]
; CHECK-NEXT:    ret i1 [[CMP]]
; CHECK:       [[IF_ELSE]]:
; CHECK-NEXT:    ret i1 false
;
  %ext = zext i16 %x to i128
  %cond = icmp ult i128 %y, %ext
  br i1 %cond, label %if.then, label %if.else

if.then:
  %conv = trunc nuw i128 %y to i16
  %cmp = icmp eq i16 %x, %conv
  ret i1 %cmp

if.else:
  ret i1 false
}

; We do not know the sign of %x, so we cannot infer nuw for %ext.
define i1 @test_icmp_ult_trunc_nsw_icmp_trunc_nuw(i64 %x, i32 %y) {
; CHECK-LABEL: define i1 @test_icmp_ult_trunc_nsw_icmp_trunc_nuw(
; CHECK-SAME: i64 [[X:%.*]], i32 [[Y:%.*]]) {
; CHECK-NEXT:    [[EXT:%.*]] = trunc nsw i64 [[X]] to i32
; CHECK-NEXT:    [[COND:%.*]] = icmp ult i32 [[Y]], [[EXT]]
; CHECK-NEXT:    br i1 [[COND]], label %[[IF_THEN:.*]], label %[[IF_ELSE:.*]]
; CHECK:       [[IF_THEN]]:
; CHECK-NEXT:    [[CONV:%.*]] = zext i32 [[Y]] to i64
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i64 [[X]], [[CONV]]
; CHECK-NEXT:    ret i1 [[CMP]]
; CHECK:       [[IF_ELSE]]:
; CHECK-NEXT:    ret i1 false
;
  %ext = trunc nsw i64 %x to i32
  %cond = icmp ult i32 %y, %ext
  br i1 %cond, label %if.then, label %if.else

if.then:
  %conv = zext i32 %y to i64
  %cmp = icmp eq i64 %x, %conv
  ret i1 %cmp

if.else:
  ret i1 false
}

declare void @cond()
