UNIT Fourier;
{ A library of useful Fourier functions, DFTs, FFTs etc}


  INTERFACE

    CONST
      F_MaxEl   = 4096;
{      F_MaxEl   = 2048;}
      F_Inverse = True;
      F_Trform  = False;
    TYPE
      Complex  = RECORD
                   rl : Single;
                   im : Single;
                 END; {Complex}
      F_Repr    = (F_ampl, F_phase);
      F_Filt    = (F_NoFilt, F_StepFilt, F_SlopeFilt, F_SlopeFilt2);
      FourArray = ARRAY [0..F_MaxEl-1] OF Complex;
    CONST
      Compl0 : Complex  = (rl:0; im:0);

    PROCEDURE DFT (VAR  Four : FourArray;   N: Integer;   Invert : Boolean);

    PROCEDURE FFT (VAR  Four : FourArray;   N: Integer;   Invert : Boolean);

    PROCEDURE F_Filter (VAR Four : FourArray;   Filt : F_Filt;   N, F_St, F_End : Integer);

    PROCEDURE Convolut (VAR  Resp, Fn: FourArray;   N: Integer;   Invert : Boolean);


  IMPLEMENTATION

    FUNCTION Arctan0 (Zlen, Ylen: Real) : Real;
        {Implements a 360 degree arctan function}
      VAR
        Ang  : Real;
      BEGIN
        IF (Ylen = 0)
           THEN IF (Zlen > 0)
                   THEN Ang := PI/2
                   ELSE Ang := -PI/2
           ELSE
             BEGIN
               Ang := ARCTAN (Zlen / Ylen);
               IF (Ylen < 0)
                  THEN Ang := Ang + PI;
             END;
        Arctan0 := Ang;
      END; {Arctan0}

    PROCEDURE DFT (VAR  Four : FourArray;   N: Integer;   Invert : Boolean);
      VAR
        H            : FourArray;
        i, k         : Integer;
        Thet         : Single;
        W,WW,dW      : Complex;
      BEGIN
        FOR i := 0 TO N-1 DO
          BEGIN
            Thet  := 2 * PI * i / N;
            IF (Invert)
               THEN Thet := - Thet;
            dW.Rl := COS (Thet);
            dW.Im := SIN (Thet);
            W.Rl  := 1;
            W.Im  := 0;
            H[i].Rl := 0;
            H[i].Im := 0;
            FOR k := 0 TO N-1 DO
              BEGIN
                H[i].Rl := H[i].Rl + Four[k].Rl * W.Rl  -  Four[k].Im * W.Im;
                H[i].Im := H[i].Im + Four[k].Rl * W.Im  +  Four[k].Im * W.Rl;
                WW   := W;  {Temporary backup}
                W.Rl := WW.Rl * dW.Rl  -  WW.Im * dW.Im;
                W.Im := WW.Rl * dW.Im  +  WW.Im * dW.Rl;
              END; {For k}
            IF (Invert)
               THEN BEGIN
                      H[i].Rl := H[i].Rl / N;
                      H[i].Im := H[i].Im / N;
                    END;
          END; {For i}
        FOR i := 0 TO N-1 DO
          Four[i] := H[i];
      END; {DFT}


    PROCEDURE BitReverse ( VAR Four: FourArray;  N: Integer );
      VAR
        a, b, b1, b2 : Integer;
        n0, n1, n2   : Byte;
        Temp         : Complex;
      BEGIN
        n0 := ROUND (LN(N) / LN(2));
        FOR a := 0 TO (N-1) DO
          BEGIN
            b  := a;
            FOR n1 := 0 TO ((n0 DIV 2) - 1) DO
              BEGIN
                n2 := n0 - n1 - 1;
                b1 := ( (b SHR n1) AND 1)  SHL n2;
                b2 := ( (b SHR n2) AND 1)  SHL n1;
                b  := b AND NOT (1 SHL n2);
                b  := b AND NOT (1 SHL n1);
                b  := b OR b1 OR b2;
              END; {For n1}
            IF (a < b)
               THEN BEGIN
                      Temp    := Four[a];
                      Four[a] := Four[b];
                      Four[b] := Temp;
                    END;
          END; {For a}
      END; {BitReverse}


    PROCEDURE FFT (VAR  Four : FourArray;   N: Integer;   Invert : Boolean);
      VAR
        i,j,k,l      : Integer;
        nn, nstep    : Integer;
        Lmax, Kmax   : Integer;
        Thet         : Single;
        W, dW, WW, Temp  : Complex;
      BEGIN
        BitReverse (Four, N);
        nn := 2;
        WHILE (nn <= N) DO
          BEGIN
            Thet  := 2 * PI / nn;
            IF (Invert)
               THEN Thet := - Thet;
            dW.Rl := COS (Thet);
            dW.Im := SIN (Thet);
            W.Rl  := 1;
            W.Im  := 0;
            nstep := nn DIV 2;
            Lmax  := N DIV nn;
            Kmax  := nn DIV 2;
            FOR k := 0 TO Kmax-1 DO
              BEGIN
                FOR l := 0 TO Lmax-1 DO
                  BEGIN
                    i := l * nn + k;
                    j := i + nstep;
                    Temp.Rl := Four[j].Rl * W.Rl  -  Four[j].Im * W.Im;
                    Temp.Im := Four[j].Rl * W.Im  +  Four[j].Im * W.Rl;
                    Four[j].Rl := Four[i].Rl - Temp.Rl;
                    Four[j].Im := Four[i].Im - Temp.Im;
                    Four[i].Rl := Four[i].Rl + Temp.Rl;
                    Four[i].Im := Four[i].Im + Temp.Im;
                  END; {For l}
                WW   := W;  {Temporary backup}
                W.Rl := WW.Rl * dW.Rl  -  WW.Im * dW.Im;
                W.Im := WW.Rl * dW.Im  +  WW.Im * dW.Rl;
              END; {For k}
            nn := nn * 2;
          END; {While}
       IF (Invert)
          THEN FOR i := 0 TO N-1 DO
                 BEGIN
                   Four[i].Rl := Four[i].Rl / N;
                   Four[i].Im := Four[i].Im / N;
                 END; {For i}
      END; {FFT}

    PROCEDURE F_Filter (VAR Four : FourArray;   Filt : F_Filt;   N, F_St, F_End : Integer);
      VAR
        i, F_l1, F_h1, F_N, N2 : Integer;
        F_damp,F_damp1,F_damp2 : single;
      BEGIN
{        Filt := F_SlopeFilt; F_N := 4;}
        F_N := 4;
        F_St  := F_St  DIV 2;
        F_End := F_End DIV 2;
        N2    := N DIV 2;
{        IF (F_St = 0) THEN F_St := 1;}
        F_l1 := TRUNC(F_St-F_N);  IF (F_l1<=0) THEN F_l1:=0;
        F_h1 := TRUNC(F_End+F_N); IF (F_h1>N2) THEN F_h1:=N2;
        CASE Filt OF
          {FL to FH Interpreted as a stop band}
          F_StepFilt : FOR i := F_St TO F_End DO
                         BEGIN  Four[i] := Compl0; Four[N-1-i] := Compl0;  END;

          {FL to FH Interpreted as a stop band}
          F_SlopeFilt : BEGIN
                          FOR i := F_l1 TO F_h1 DO
                            BEGIN
{                              F_damp1 := 1.0/sqrt(1+(i-F_l1)/F_N);
                              F_damp2 := 1.0/sqrt(1+(F_h1-i)/F_N);}
{                              F_damp1 := 1.0/(1+(i-F_l1)/F_N);
                              F_damp2 := 1.0/(1+(F_h1-i)/F_N);}
                              F_damp1 := exp(-(i-F_l1)*0.01) ;
                              F_damp2 := exp(-(F_h1-i)*0.01) ;

                              IF (F_st<=1)   THEN F_damp1:=0.0;
                              IF (F_end>=N2) THEN F_damp2:=0.0;
                              F_damp:=F_damp1+F_damp2; IF (F_damp>1) THEN F_damp:=1;
                              Four[i].Rl := Four[i].Rl*F_damp;
                              Four[i].Im := Four[i].Im*F_damp;
                              Four[N-1-i].Rl := Four[N-1-i].Rl*F_damp;
                              Four[N-1-i].Im := Four[N-1-i].Im*F_damp;
                            END;
                        END;

          {FL to FH Interpreted as a pass band}
          F_SlopeFilt2 : BEGIN
                          FOR i := 0 TO F_St DO
                            BEGIN
                              F_damp := exp(-(F_St-i)*0.01) ;
                              Four[i].Rl := Four[i].Rl*F_damp;
                              Four[i].Im := Four[i].Im*F_damp;
                              Four[N-1-i].Rl := Four[N-1-i].Rl*F_damp;
                              Four[N-1-i].Im := Four[N-1-i].Im*F_damp;
                            END;
                          FOR i := F_End TO ((N DIV 2) -1) DO
                            BEGIN
                              F_damp := exp(-(i-F_End)*0.01) ;
                              Four[i].Rl := Four[i].Rl*F_damp;
                              Four[i].Im := Four[i].Im*F_damp;
                              Four[N-1-i].Rl := Four[N-1-i].Rl*F_damp;
                              Four[N-1-i].Im := Four[N-1-i].Im*F_damp;
                            END;
                        END;

        END; {Case}
      END; {F_Filter}

    PROCEDURE Convolut (VAR  Resp, Fn: FourArray;   N: Integer;   Invert : Boolean);
      VAR
        i     : Integer;
      BEGIN
        FFT (Resp, N, F_Trform);
        FFT (Fn,   N, F_Trform);
        IF (Invert)
           THEN FOR i := 1 TO N DO
                  BEGIN
{                    Fn[i] := Fn[i] / Resp[i]}
                  END
           ELSE FOR i := 0 TO N-1 DO
                  BEGIN
                    Fn[i].rl := Fn[i].rl * Resp[i].rl - Fn[i].im * Resp[i].im;
                    Fn[i].im := Fn[i].rl * Resp[i].im + Fn[i].im * Resp[i].rl;
                  END;
        FFT (Fn, N, F_Inverse);
      END; {Convolut}


  BEGIN {Fourier Unit - Initialization}
  END. {Fourier Unit}
