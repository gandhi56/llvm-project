! Offload test that checks an allocatable array can be mapped alongside a
! non-allocatable derived type when both are contained within a derived type
! can be mapped correctly via member mapping and then written to.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    type :: bottom_layer
    real(4) :: i
    integer(4) :: array_i(10)
    integer(4) :: k
    end type bottom_layer

    type :: top_layer
    real(4) :: i
    integer, allocatable :: scalar
    integer(4) :: array_i(10)
    real(4) :: j
    integer, allocatable :: array_j(:)
    integer(4) :: k
    type(bottom_layer) :: nest
    end type top_layer

    type(top_layer) :: one_l

    allocate(one_l%array_j(10))
    allocate(one_l%scalar)

    do i = 1, 10
        one_l%nest%array_i(i) = i
    end do

!$omp target map(tofrom: one_l%nest, one_l%array_j)
    do i = 1, 10
        one_l%array_j(i) = one_l%nest%array_i(i) + i
    end do
!$omp end target

    print *, one_l%nest%array_i
    print *, one_l%array_j
end program main

!CHECK: 1 2 3 4 5 6 7 8 9 10
!CHECK: 2 4 6 8 10 12 14 16 18 20
