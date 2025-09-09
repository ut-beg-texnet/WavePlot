UNIT Funct;

{A Collection of trivial but useful functions to complement library}


  INTERFACE

     CONST
       Ln10 : Single = 2.302585125;
       LargeSingle : Single = 1e30;
     TYPE
       Matrix3    = ARRAY[1..3,1..3] OF Single;

     FUNCTION Max (Var1, Var2 : Single) : Single;

     FUNCTION Min (Var1, Var2 : Single) : Single;

     FUNCTION iMax (Var1, Var2 : Integer) : Integer;

     FUNCTION iMin (Var1, Var2 : Integer) : Integer;

     FUNCTION LOG10 (Val:Single): Single;

     FUNCTION POW (a, n : single): Single;

     PROCEDURE IncrStep (VAR Val : Word;
                             Low, High : Word);

     PROCEDURE DecrStep (VAR Val : Word;
                             Low, High : Word);

     FUNCTION Arctan0 (Zlen, Ylen: Single) : Single;

     FUNCTION Ival (ValStr : String) : Integer;

     FUNCTION DecDigits (Value: Single; Signif: Byte) : Byte;

     FUNCTION Strng (Value : Single; Control: Word) : String;

     FUNCTION RStrng (i1, i2 : Integer) : String;

     FUNCTION RatioStr (Ratio: Single) : String;

     PROCEDURE SwapVal (VAR  Var1, Var2 : Single);

     PROCEDURE RotateCoords (VAR  V1, V2: Single;  Ang : Single);

     FUNCTION RoundPlaces (Value: Single; Rdig: Word) : Single;

     FUNCTION IROUND (x : Single): LongInt;

     FUNCTION EXP10 (Pow : Integer): Single;

     PROCEDURE CubeRoots (aa,bb,cc,dd: single;
                          VAR   r1,r2,r3,r1i,r2i,r3i  : single   );

     PROCEDURE Mat_I3 (VAR m: Matrix3);

     PROCEDURE Mat_Rot3D (VAR m: Matrix3; ax:Word; Ang_d:single);


  IMPLEMENTATION

     FUNCTION Max (Var1, Var2 : Single) : Single;
       BEGIN
         IF (Var1 >= Var2)
            THEN Max := Var1
            ELSE Max := Var2;
       END; {Max}

     FUNCTION Min (Var1, Var2 : Single) : Single;
       BEGIN
         IF (Var1 <= Var2)
            THEN Min := Var1
            ELSE Min := Var2;
       END; {Min}

     FUNCTION iMax (Var1, Var2 : Integer) : Integer;
       BEGIN
         IF (Var1 >= Var2)
            THEN iMax := Var1
            ELSE iMax := Var2;
       END; {iMax}

     FUNCTION iMin (Var1, Var2 : Integer) : Integer;
       BEGIN
         IF (Var1 <= Var2)
            THEN iMin := Var1
            ELSE iMin := Var2;
       END; {iMin}

{     FUNCTION Log10 (x : Single) : Single;
       BEGIN
         Log10 := LN(x) / Ln10;
       END; {Log10}

     PROCEDURE IncrStep (VAR Val : Word;
                             Low, High : Word);
       BEGIN
         IF (Val = High)
            THEN Val := Low
            ELSE Inc (Val);
       END; {IncrStep}

     PROCEDURE DecrStep (VAR Val : Word;
                             Low, High : Word);
       BEGIN
         IF (Val = Low)
            THEN Val := High
            ELSE Dec (Val);
       END; {DecrStep}

     FUNCTION Arctan0 (Zlen, Ylen: Single) : Single;
       VAR
         Ang  : Single;
       BEGIN
         IF (Ylen = 0)
            THEN IF (Zlen > 0)
                    THEN Ang := PI/2
                    ELSE Ang := -PI/2
            ELSE
              BEGIN
                Ang := ARCTAN (Zlen/Ylen);
                IF (Ylen < 0)
                   THEN IF (Zlen > 0)
                           THEN Ang := Ang + PI
                           ELSE Ang := Ang - PI;
              END;
         Arctan0 := Ang;
       END; {Arctan0}

     FUNCTION Ival (ValStr : String) : Integer;
       VAR
         ValTmp : Single;
         ErrPos : Integer;
       BEGIN
         VAL (ValStr, ValTmp, ErrPos);
         Ival := ROUND (ValTmp);
       END; {Ival}

     PROCEDURE SwapVal (VAR  Var1, Var2 : Single);
       VAR
         Temp : Single;
       BEGIN
         Temp := Var1;
         Var1 := Var2;
         Var2 := Temp;
       END; {SwapVal}


     FUNCTION DecDigits (Value: Single; Signif: Byte) : Byte;
         {Signif is the number of significant digits. The function
                     returns the number of decimal digits required}
       VAR
         Dgts : Single;
       BEGIN
         Value := ABS(Value);
         IF (Value = 0)
            THEN Dgts := Signif - 1
            ELSE BEGIN
                   Dgts := Ln(Value) / Ln10;
                   IF (Dgts < 0)
                      THEN IF ( ABS(FRAC(Dgts)) >  0)
                              THEN Dgts := Dgts - 1; {To counteract the effects of truncating a negative number}
                   Dgts := Signif - TRUNC(Dgts) - 1;
                 END;
         IF (Dgts < 0)
            THEN Dgts := 0;
         DecDigits := TRUNC (Dgts);
       END; {DecDigits}


     FUNCTION Strng (Value : Single; Control: Word) : String;
         {The control word contains two bytes - MSB indicates the minimum number of significant digits,
              while the LSB indicates the minimum number of digits after the decimal point}
       VAR
         S, S1           : String;
         DecPlaces       : Byte;
         SigDigits       : Byte;
         Len, SExp, NExp : LongInt;
         Shift           : LongInt;
         DecPos          : LongInt;
         ErrCode         : Integer;
       BEGIN
         DecPlaces := Lo(Control);
         SigDigits := Hi(Control);
         IF (SigDigits < $F0)
            THEN BEGIN
                   IF (SigDigits > 0)
                      THEN DecPlaces := TRUNC ( Max (DecPlaces, DecDigits(Value, SigDigits) )  );
                   STR (Value:1:DecPlaces, S);
                 END
            ELSE IF (DecPlaces > $00)
                    THEN BEGIN
                             { DecPlaces = digits in exponent;  SigDigits (-$F0) = digits in mantissa. }
                           IF (DecPlaces > 3)
                              THEN DecPlaces := 2;
                           STR (Value:(8+(SigDigits-$F0)), S);
                           len    := Length (S);
                           VAL ( Copy(S, len-4, 5), SExp, ErrCode);
                           IF (SExp = 0)
                              THEN delete (S, len-5, 6)
                              ELSE delete (S, len-3, 4-DecPlaces);
                         END
                    ELSE BEGIN
                             { SigDigits (-$F0) = digits in mantissa. }
                             { Exponent is normalized to a multiple of 3. }
                           STR (Value:14, S);
                           len    := Length (S);
                           VAL ( Copy(S, len-4, 5), SExp, ErrCode);
                           delete (S, len-5, 6);
                           Shift  := SExp MOD 3;
                           IF (Shift < 0)
                              THEN Shift := 3 + Shift;
                           NExp   := SExp - Shift;
                           STR (NExp:3, S1);
{                           IF (S1[1] = ' ') THEN S1[1] := S1[2];}
                           IF (S1[2] IN [' ', '+', '-']) THEN S1[1] := S1[2];
                           IF (S1[1] = ' ') THEN S1[1] := '+';
                           IF (S1[2] IN [' ', '+', '-']) THEN S1[2] := '0';

                           DecPos := Pos ('.', S);
                           delete (S, DecPos,1);
                           insert ('.', S, DecPos+Shift);
                           SigDigits := SigDigits - $F0 + 2;
                           IF (SigDigits < 5)
                              THEN SigDigits := 5;
                           delete (S, SigDigits, len-SigDigits);
                           IF (NExp <> 0) THEN S := S + 'E' + S1;
                           IF (S[1] = '.') THEN insert ('0', S, 1);
                         END;
         Strng := S;
       END; {Strng}


     FUNCTION Strng0 ( Value : Single;
                       Stype : Word; SigDig, MaxDec, ExpSz : Word ) : String;

        { Stype : 1 = Decimal, unless MaxDec exceeded;
                  2 = Exponent - normalized form: n.nnnEdd;
                  3 = Exponent MOD 3 - S.I.
                  4 = Best Choice;
          SigDig = Significant Digits;
          MaxDec = Maximum decimal digits, before showing in exponential form;
          ExpSz  = Exponent Size; }

       VAR
         S, SMant, SExp  : String;
         DecPlaces       : Byte;
         SigDigits       : Byte;
         Len, VExp       : Integer;
         ErrCode  : Integer;
         DecPos          : Word;

       BEGIN
         IF (ExpSz < 1) THEN ExpSz := 1;
         IF (ExpSz > 4) THEN ExpSz := 4;

         STR (Value:(8+SigDigits), S);
         len    := Length (S);
         SMant  := Copy(S, 1, len-6);
         SExp   := Copy(S, len-4, 5);
         delete (SExp, 2, 4-ExpSz);
         VAL (SExp, VExp, ErrCode);

         IF (Stype = 4)
            THEN IF (MaxDec < (SigDig+2+ExpSz+2) )
                    THEN MaxDec := (SigDig+2+ExpSz+2);
         IF (Stype = 4) THEN Stype := 1;

         IF (Stype = 1)
            THEN IF ( (VExp < -(MaxDec-SigDig+1) ) OR (VExp > (MaxDec-1) ) )
                    THEN Stype := 2;

{         CASE Stype OF
           1 : BEGIN
                 Shift (SMant, VExp);
               END;
           2 : BEGIN
                 S := Smant + 'E' + SExp;
               END;
           3 : BEGIN
                 Shft  := SExp MOD 3;
                 IF (Shft < 0)
                    THEN Shft := 3 + Shft;
                 Shift (SMant, Shft)
                 NExp   := SExp - Shft;
                 IF (NExp <> 0) THEN S := SMant + 'E' + SExp;
                 IF (S[1] = '.') THEN insert ('0', S, 1);
               END;
         END; {Case}

         Strng0 := S;
       END; {Strng0}

     FUNCTION RStrng (i1, i2 : Integer) : String;
       BEGIN
         RStrng := '('+strng(i1,$0000)+','+strng(i2,$0000)+')';
       END; {RStrng}

     PROCEDURE RotateCoords (VAR  V1, V2 : Single;  Ang : Single);
          {Rotates a vector in one coordinate system through a C.W. angle to the new coordinate system}
       VAR
         Temp1, Temp2 : Single;
       BEGIN
         Temp1 :=  V1 * Cos(Ang) + V2 * Sin (Ang);
         Temp2 := -V1 * Sin(Ang) + V2 * Cos (Ang);
         V1 := Temp1;
         V2 := Temp2;
       END; {RotateCoords}

     FUNCTION RatioStr (Ratio: Single) : String;
       { Takes a decimal value and returns an Integer Ratio in a String}
       VAR
         Num, Denom     : Integer;
         Mult           : Integer;
         NumStr, DenStr : String;
       BEGIN
         Ratio := ABS(Ratio);
         Denom := 1;
         WHILE ( FRAC(Ratio)*10 > ROUND(Ratio) ) DO
           BEGIN
             Mult  := ROUND ( 1/FRAC(Ratio) );
             IF (Mult = 1)
                THEN Mult := 10;
             Ratio := Ratio * Mult;
             Denom := Denom * Mult;
           END;
         Num := ROUND(Ratio);
         STR (Num:1, NumStr);
         STR (Denom:1, DenStr);
         RatioStr := NumStr + ' : ' + DenStr;
       END; {RatioStr}

     FUNCTION RoundPlaces (Value: Single; Rdig: Word) : Single;
       VAR
         Dig   : Integer;
         Divd  : Single;
         sign  : Integer;
       BEGIN
         IF (Value = 0)
            THEN RoundPlaces := Value
            ELSE BEGIN
                   sign  := ROUND(Value/abs(Value));
                   Value := ABS(Value);
                   Dig   := TRUNC (LN(Value)/Ln10) - Rdig;
                   Divd  := EXP(Dig*Ln10);
                   Value := TRUNC (Value / Divd + 0.99);
{                   Value := TRUNC (Value / Divd + 0.999999*EXP(-(Rdig-1)*Ln10));}
                   RoundPlaces := Sign * (Value * Divd);
                 END;
       END; {RoundPlaces}

     FUNCTION IROUND (x : Single): LongInt;
        {OverRides the std Round function - Ensures that large value
           does not change sign during conversion to integer}
       CONST
         MaxInt  = 16383; {Normally 32767}
       BEGIN
         IF (x > MaxInt)
            THEN x := MaxInt
            ELSE IF (x < -MaxInt)
                    THEN x := -MaxInt;
         iROUND := ROUND(x);
       END; {iROUND}

     FUNCTION EXP10 (Pow : Integer): Single;
        {Returns 10 to the power of i}
       VAR
         i : Integer;
         Val : Single;
       BEGIN
         Val := 1;
         IF (Pow >= 0)
            THEN FOR i := 1 TO Pow DO
                   Val := Val * 10
            ELSE FOR i := Pow TO -1 DO
                   Val := Val * 0.1;
         EXP10 := Val;
       END; {EXP10}

     FUNCTION LOG10 (Val:Single): Single;
        {Returns log to base 10, but checks if close to an integer}
       VAR
         logval : Single;
       BEGIN
         logval := LN(Val) / Ln10;
         IF (  ABS (logval - ROUND (logval) )  < 0.004 )
            THEN logval := ROUND (logval);
         LOG10 := logval;
       END; {LOG10}

     FUNCTION POW (a, n : single): Single;
        {Returns a to the power of p}
       VAR
         sign, nn  : Integer;
         val       : Single;
       BEGIN
         val := 0;
         IF (a<>0) THEN
            BEGIN
              sign := 1;
              val  := Exp(Ln(abs(a))* n);
              IF (round(n) = trunc(n))
                THEN
                   BEGIN
                     IF (odd(round(n)) ) THEN
                        sign  := ROUND(a/abs(a));
                   END
                ELSE IF (round(1/n) = trunc(1/n)) THEN
                        BEGIN
                          nn := round(1/n);
                          IF (odd(round(nn)) ) THEN
                             sign  := ROUND(a/abs(a));
                        END;
              val := val*sign;
            END;
         POW := Val;
       END; {POW}


     PROCEDURE CubeRoots (aa,bb,cc,dd: single;
                          VAR   r1,r2,r3,r1i,r2i,r3i  : single   );
          {Uses analytical expression to solve roots of a cubic}
       VAR
         b,c,d, v1,v2,v3,v4, cu2, qb,qc,qd, sq : Double;
         v4_m,v4_a,v5_m,v5_a,v6_m,v6_a,v6_r,v6_i,v7_m,v7_a : Double;
       BEGIN
         cu2 := exp(ln(2)/3);
         r1 := 1e20; r2 := 1e20; r3 := 1e20;
         r1i := 0; r2i := 0; r3i := 0;
         b:=bb/aa; c:=cc/aa; d:=dd/aa;
         v1 := -b*b+3*c;
         v2 := -2*b*b*b +9*b*c -27*d;
         v3 := 4*v1*v1*v1 + sqr(v2);

         IF (v3>=0)
            THEN BEGIN  {v3 is Real}
                   v3 := sqrt(v3);
                   v4 := (v2+v3);
                   IF (v4>0)
                      THEN v4 := exp (ln(v4)/3)
                      ELSE IF (v4<0)
                              THEN BEGIN v4 := -exp (ln(-v4)/3); END;
                   IF (V4<>0)
                      THEN r1 := (-b - cu2*v1/v4 +v4/cu2)/3
                      ELSE IF (v1=0)
                              THEN r1 := (-b +v4/cu2)/3
                              ELSE EXIT;
                 END {Real v3}

            ELSE BEGIN  {v3 is Imaginary}
                   v3 := sqrt(-v3);
                   v4_m := sqrt(v2*v2+v3*v3);
                   v4_a := arctan0 (v3,v2);
                   IF (v4_m>0)
                      THEN v4_m := exp (ln(v4_m)/3)
                      ELSE IF (v4_m<0)
                              THEN BEGIN v4_m := -exp (ln(-v4_m)/3); END;
                   v4_a := v4_a/3;
                   v5_m := sqr(v4_m);
                   v5_a := v4_a*2;
                   v6_r := -cu2*v1 + v5_m*cos(v5_a) / cu2;
                   v6_i :=           v5_m*sin(v5_a) / cu2;
                   v6_m := sqrt(sqr(v6_r) + sqr(v6_i));
                   v6_a := arctan0 (v6_i,v6_r);
                   v7_m := v6_m / v4_m;
                   v7_a := v6_a - v4_a;
                   r1   := (-b + v7_m*cos(v7_a))/3;
                   r1i  := (     v7_m*sin(v7_a))/3;
                 END; {v3 Imaginary}

         {now solve quadratic}
         qb := b+r1;
         qc := c+r1*qb;
         qd := -r1*qc;    {test, should be d}
{         writeln ('cubic: ', b:12, c:12, d:12);
         writeln ('quad:  ',qb:12, qc:12, qd:12);}
         sq := (qb*qb - 4*qc);
         IF (sq>=0)
            THEN BEGIN  sq:= Sqrt(sq);  r2:=(-qb+sq)*0.5;  r3:=(-qb-sq)*0.5;  END
            ELSE BEGIN  sq:= Sqrt(-sq); r2:=-qb*0.5; r3:=r2; r2i:=sq*0.5; r2i:=-sq*0.5;  END;

{         writeln ('roots: ',r1:12, r2:12, r3:12, r1i:11, r2i:11, r3i:11);}
       END; {CubeRoots}


    {** Matrix Routines **}
     PROCEDURE Mat_I3 (VAR m: Matrix3);
       VAR
         i,j : Integer;
       BEGIN
         FOR i:=1 TO 3 DO
          FOR j:=1 TO 3 DO
            IF (i=j) THEN m[i,j]:=1
                     ELSE m[i,j]:=0;
       END; {Mat_I3}

     PROCEDURE Mat_Rot3D (VAR m: Matrix3; ax:Word; Ang_d:single);
       VAR
         i,j         : Integer;
         Ang, CA, SA : single;
         r,t         : Matrix3;
       BEGIN
         Ang := ang_d * pi/180;  CA := cos(Ang);  SA := sin(Ang);
         CASE ax OF
           1 : BEGIN
                 r[1,1]:=1;   r[1,2]:=0;   r[1,3]:=0;
                 r[2,1]:=0;   r[2,2]:=CA;  r[2,3]:=SA;
                 r[3,1]:=0;   r[3,2]:=-SA; r[3,3]:=CA;
               END;
           2 : BEGIN
                 r[1,1]:=CA;  r[1,2]:=0;   r[1,3]:=SA;
                 r[2,1]:=0;   r[2,2]:=1;   r[2,3]:=0;
                 r[3,1]:=-SA; r[3,2]:=0;   r[3,3]:=CA;
               END;
           3 : BEGIN
                 r[1,1]:=CA;  r[1,2]:=SA;  r[1,3]:=0;
                 r[2,1]:=-SA; r[2,2]:=CA;  r[2,3]:=0;
                 r[3,1]:=0;   r[3,2]:=0;   r[3,3]:=1;
               END;
           END;
         t := m;
{         FOR i:=1 TO 3 DO
          FOR j:=1 TO 3 DO
            t[i,j] := m[i,j];}
         FOR i:=1 TO 3 DO
          FOR j:=1 TO 3 DO
            m[i,j] := r[i,1]*t[1,j]+r[i,2]*t[2,j]+r[i,3]*t[3,j];
       END; {Mat_Rot3D}


END. {Funct}
