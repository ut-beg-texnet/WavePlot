
  PROCEDURE InterpolSnap;
      {Plots interpolated blocks}
    CONST
      ColPos : ARRAY [0..3] OF Byte = (8,4,2,1);
      BitPos : ARRAY [1..8] OF Byte = (1,2,4,8,16,32,64,128);
    TYPE
      IBuf  = ARRAY [1..$2000] OF Byte;
    VAR
      Buf,Bf2      : IBuf;
      i,j          : Word;
      I2, J2       : Word;
      Lev          : Single;
      C            : ARRAY [0..1,0..1] OF Single;
      xPix, yPix   : Pixels;
      ix, iy, Yc   : Integer;
      Bl, St       : Pixels; {Block Size - size ofone element in pixels}
      Col, Pos, Bit, jPos, iStp, jStp : Word;
      k, Sz        : Word;

    BEGIN {InterpolSnap}
      CalcLevels (CurSnap);
      With SnapMap [CurSnap] DO
        Ds_GetBlockSz (QuickX-1, QuickY-1, Bl, St);
      xPix[1] := St[1];
      FOR ix := 1 TO SnapMap [CurSnap].QuickX-1 DO
        BEGIN
          xPix[2] := xPix[1] + Bl[1] - 1;
          yPix[1] := St[2];
          FOR iy := 1 TO SnapMap [CurSnap].QuickY-1 DO
            BEGIN
              Yc      := SnapMap [CurSnap].QuickY+1 - iy;
              yPix[2] := yPix[1] + Bl[2] - 1;
              C[0,0] := ( SnapShot^[ix  ,Yc  ] );
              C[1,0] := ( SnapShot^[ix+1,Yc  ] );
              C[0,1] := ( SnapShot^[ix  ,Yc-1] );
              C[1,1] := ( SnapShot^[ix+1,Yc-1] );
              I2 := Bl[1];
              J2 := Bl[2];
              Buf[1] := Bl[1];
              Buf[2] := 0;
              Buf[3] := Bl[2];
              Buf[4] := 0;
              istp := TRUNC ( (I2-1)/8) + 1;
              jstp := istp*4;
              FOR j := 1 TO J2 DO
                FOR i := 1 TO I2 DO
                  BEGIN
                    Lev := (  (C[0,0]*(J2-j) + C[0,1]*j) * (I2-i)  + (C[1,0]*(J2-j) + C[1,1]*j) * i  ) / I2/ J2;
{                    Buf[4+(j-1)*I2+i] := LevCol [GetLevel(Lev)];}
                    PutPixel (xPix[1]+i-1, yPix[1]+j-1, LevCol [GetLevel (Lev)] );

{                    PutPixel (xPix[1]+i-1, yPix[1]+j-1, iy);
                    Lev := (  (C[0,0]*(J2-j) + C[0,1]*j) * (I2-i)  + (C[1,0]*(J2-j) + C[1,1]*j) * i  ) / I2/ J2;
                    PutPixel (xPix[1]+i-1, yPix[1]+j-1, LevCol [GetLevel (Lev)] );
                    Col := LevCol [GetLevel(Lev)];
                    Col := iy;
                    jPos := 4+1+(j-1)*jstp;
                    Pos := jPos + (i-1) DIV 8;
                    Bit := BitPos[8 - ( (i-1) MOD 8)];
                    FOR k := 0 TO 3 DO
                      BEGIN
                        IF (Bit = 128)
                           THEN Buf[Pos+k*istp] := 0;
                        IF ( (Col AND ColPos[k]) > 0)
                           THEN Buf[Pos+k*istp] := Buf[Pos+k*istp] OR Bit;
                      END;}
                  END;
{                  Sz := ImageSize (xPix[1], yPix[1], xPix[2], yPix[2]);
                  GetImage (xPix[1], yPix[1], xPix[2], yPix[2], Bf2);
                  PutImage (xPix[1], yPix[1], Buf, NormalPut);}
              yPix[1] := yPix[2] + 1;
            END;
          xPix[1] := xPix[2] + 1;
        END;
    END; {InterpolSnap}

