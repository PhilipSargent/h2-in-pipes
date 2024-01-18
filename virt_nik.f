! Converted using https://convertio.co/ocr/ from the image of the code inside 
! 'Virtual Nikuradse' full version on : https://www.researchgate.net/publication/232974572
! transcribed by Philip Sargent, 2024-01-18

IMPLICIT NONE

INTEGER :: I, J, K 
INTEGER :: INDEX J

LOGICAL :: Inner_Loop, Outer_Loop

REAL :: S(4), T(4)
REAL :: M(5), N(5)

DOUBLE PRECISION :: A (5), B(5), R (4)
DOUBLE PRECISION :: AA(5,6), BB(5,6), RR(5,6) AAX(5,6), BBX (5, 6), RRX(5, 6) P(5), F(5)
DOUBLE PRECISION :: PP(5,6), FF(5,6)
DOUBLE PRECISION :: AA_5_FL, AA_5_FR Re, Sigma
DOUBLE PRECISION :: Lamda, Lamda_S, Lamda_R

WRITE (*, *)

PRINT*, ’ Welcome! This code is for calculating Lamda = f(Re, Sigma)’ 
PRINT*, ’ for pipe flows using the Virtual Nikuradse Correlation’ 
PRINT*, ’ published in Journal of Turbulence (2009). The Virtual’ 
PRINT*, ’ Nikuradse Correlation (VNC) works for the flow in smooth’ 
PRINT*, ’ and effectively smooth pipes and the flow in rough pipes’ 
PRINT*, ’ with pasted sand-grain roughness. VNC can be used for any’ 
PRINT*, ’ Reynolds numbers and all fluid flow regimes including’ 
PRINT*, ’ laminar, transition and turbulent flows. The roughness’ 
PRINT*, ’ ratio is defined by Sigma = a/k, where a is the radius of’ 
PRINT*, ’ pipe and k is the average projection. When the Reynolds’ 
PRINT*, ’ number (Re) and roughness ratio (Sigma) are given, the’ 
PRINT*, ’ friction factor (Lamda) can be easily determined below. ’
WRITE (*, *)
PRINT*, ’ Reminder: Users must know that VNC requires Re>0 and &
Sigma>=15. ’
WRITE (*, *)
Outer_Loop=. FALSE.

DO
    PRINT*, ’-----------------------------------------------------------&
    J
    PRINT*, ’ Please input the value of flow Reynolds number (Re>0) &
    then press ENTER:’
    READ*, Re
    PRINT*, ’ Please input the value of roughness ratio (Sigma>=15) &
    then press ENTER:’
    READ*, Sigma
    S= (/—50., -15.,	-5., -2./)
    T= (/ 0.5, 0.5,	0.5, 0.5/)
    
    A=(/64., 0.000083, 0.3164, 0.1537, 0.0753/) B= (/-1., 0.75, -0.25, -0.185, -0.136/)
    R=(/2320., 3810., 70000.	, 2000000./)				
    M= (/—2., -5., -5., -5.,	-5. /)				
    N=(/0.5, 0.5, 0.5, 0.5,	0. 5/)				
    AAX=reshape((/0. 05016,	0. 0476,	0. 00944,	0. 02076,	0. 00253,	&
        0. 03599,	0.0331,	0. 00758,	0. 02448,	0. 0225,	&
        0. 02615,	0. 0235,	0.00615,	0. 02869,	0. 0561,	&
        0.01851,	0.0161,	0.00491,	0. 03410,	0. 1031,	&
        0.01344,	0.0113,	0. 00397,	0. 04000,	0. 1307,	&
        0. 00965,	0. 0079,	0. 00320,	0. 04710,	0. 1593/), &	
        (/5, 6/))					
    BBX=reshape((/ 0.0,	0. 002,	0. 1229,	0. 2945,	0. 5435,	&
        0. 0,	0. 002,	0. 1,	0. 2413,	0. 2687,	&
        0. 0,	0. 002,	0. 0822,	0. 2003,	0. 1417,	&
        0. 0,	0. 002,	0. 0665,	0. 1619,	0. 0693,	&
        0. 0,	0. 002,	0. 0544,	0. 1337,	0. 0356,	&
        0. 0,	0. 002,	0. 0445,	0. 1099,	0. 0181/), &	
        (/5, 6/))					
    RRX=reshape((/1010000.,	23900.,	6000.,	6000.,	1289.,	&
        1400000.,	49800.,	12300.,	10280.,	3109.,	&
        1900000.,	100100.,	23900.,	17100.,	7109.,	&
        2660000.,	214500.,	50100.,	29900.,	18109.,	&
        3650000.,	441000.,	99900.,	50000.,	42109.,	&
        5000000.,	910000.,	200000.,	85070.,	100109./),*	
        (/5, 6/))
    AA(1, :)=AAX(l, :)+0. 0098 
    AA(2, :) =AAX (2, :)+0.011 
    AA(3, :)=AAX(3, :)+0. 0053 
    AA(4, :)=AAX(4, :)
    AA(5, :)=AAX(5, :)
    BB(1, :)=BBX(1, :)
    BB (2, :)=BBX(2, :)
    BB (3, :)=BBX(3, :)+0.015 
    BB (4, :)=BBX (4, :)-0. 191 
    BB (5, :)=BBX(5, :)-0. 2032
    RR(1, :)=RRX(l, :)
    RR(2, :)=RRX(2, :)
    RR (3, :)=RRX(3, :)
    RR (4, :)=RRX(4, :)
    RR(5, :)=RRX(5, :) +1891.
    AA(l, :)=0. 17805185*(Sigma**(-0. 46785053))+0. 0098 
    AA(2,:)=0. 18954211*(Sigma**(-0. 51003100))+0. 011 
    AA(3, :)=0. 02166401*(Sigma**(-0. 30702955))+0. 0053 
    AA(4,:)=0.01105244*(Sigma**( 0.23275646)) 
    AA_5_FL=0. 00255391*(Sigma**( 0.8353877 ))-0.022 
    AA_5_FR=0. 92820419*(Sigma**( 0. 03569244))-1.
    AA(5, :)=LDFA(Sigma, AA 5 FL, AA 5 FR, -50., 0. 5, 93. DO)
    BB(1, :)=0.0 BB(2, :)=0. 002
    BB(3,:)=0.26827956*(Sigma**(-0. 28852025))+0. 015 BB(4,:)=0.62935712*(Sigma**(-0. 28022284))-0. 191 BB(5,:)=7.3482780 *(Sigma**(-0. 96433953))-0. 2032
    RR(1,:)=295530. 05 *(Sigma**( 0.45435343))
    RR(2, :)=1451. 4594 *(Sigma**( 1.0337774 ))
    RR(3, :)=406.33954 *(Sigma**( 0.99543306))
    RR(4, :)=783. 39696 *(Sigma**( 0.75245644))
    RR(5, :)=45. 196502 *(Sigma**( 1.2369807 ))+1891.
    
    DO 1=1,5
    P (I) =A (I) * (Re**B (I))
    END DO
    
    F (1) =P (1)
    
    DO 1=1,4
    F(I+l)=LDFA(Re,F(I),P(I+l),S(I),T(I),R(I))
    END DO
    
    DO 1=1,5 
    DO J=l, 6
    PP(I, J)=AA(I, J)*(Re**BB(I, J))
    END DO 
    END DO
    
    FF(1, :)=PP(1, :)
    
    DO 1=1,4 
    DO J=l, 6
    FF (1+1, J) =LDFA (Re, PP (1+1, J), FF (I, J), M (I), N (I), RR (I, J)) 
    END DO 
    END DO
    
    INDEX_J=1
!   INDEX_J is roughness index and may be 1, 2, 3, 4, 5, or 6 for 
!   the six values of roughness in Nikuradse’s (1933) data. The 
!   result does not depend on INDEX_J.

    Lamda_S=F(5)
    Lamda_R=FF(5, INDEX_J)
    Lamda=LDFA(Re, Lamda_S, Lamda_R, M(5), N(5), RR(5, INDEX_J))
    
    WRITE (*, *)
    PRINT*, ’The friction factor is: Lamda = ’, Lamda WRITE (*, *)
    PRINT*, ’ Do you want to calculate another one?’
    
    DO
        PRINT*, ’[Press "1" to continue or press "0" to exit]’
        READ*, K WRITE (*, *)
        
        IF (K==0) THEN Outer_Loop=. TRUE.
        Inner_Loop=. TRUE.
        ELSE IF (K==l) THEN Outer_Loop=. FALSE.
        Inner_Loop=. TRUE.
        ELSE
        PRINT*, ’ERROR!!! Please only input the number 1 or 0!’ Inner_Loop=. FALSE.
        END IF
        IF (Inner_Loop) EXIT 
     END DO
     IF (Outer_Loop) EXIT
     END DO
     
WRITE (*, *)
PRINT*, ’---------------------------------------------------------&
PRINT*, ’Thank you for using the VNC friction factor calculator.’ 
PRINT*, ’If there is any question, please feel free to contact me 
PRINT*, ’at yhaoping@aem.umn.edu or Haoping.Yang@noaa.gov.’
WRITE (*, *)
WRITE (*, *)

CONTAINS
LOGISTIC DOSE FUNCTION ALGORITHM
FUNCTION LDFA(Var_Independent, FL, FR, POWER.SM, POWERJTN, RCR)

IMPLICIT NONE

REAL :: POWER_SM, POWERJTN

DOUBLE PRECISION :: Var_Independent 
DOUBLE PRECISION :: FL, FR 
DOUBLE PRECISION :: RCR 
DOUBLE PRECISION :: LDFA

LDFA=FL+(FR-FL)/((1. +(Var_Independent/RCR)**POWER_SM)**POWER_TN) 
END FUNCTION LDFA

END PROGRAM Virtual Nikuradse Correlation