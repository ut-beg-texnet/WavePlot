  {Part of Dspl Unit}
   {Snapshots / 2-D graphing}


  PROCEDURE CalcPix (VAR Hz_st,Hz_blk,Hz_fin, Vt_st,Vt_blk,Vt_fin: LongInt;
                     VAR Vt_sgn: Integer);
    {Calculates integer block sizes and positions for fastest calculation
         - still need to be divided by Xlen or Ylen during use}
      {blk=block size; st=first pix; fin=last pix;}
      {sgn=+/-1 depending on sign of scale (Ypx) which may be negative}
      {Vpos=Variable position in WAVE grid;}
    VAR
      Xpos, Ypos         : LongInt;    {Extra offsets due to variable position}
    BEGIN {CalcPix}
      WITH CurSnapRec DO
       WITH Scale2D DO
        BEGIN
          Vt_sgn  := ROUND(Ypx/ABS(Ypx));
          Hz_blk  := Xstp * Xpx;
          Vt_blk  := Ystp * Ypx;
          Xpos := (Vpos MOD 2) * Xpx DIV 2;
          Ypos := (Vpos DIV 2) * Ypx DIV 2;
          Hz_St  := Xoff*Xlen + Xpos + Xpx*(X1-XX1);
          Vt_St  := Yoff*Ylen + Ypos + Ypx*(Y1-YY1);
          Hz_fin := Xoff*Xlen + Xpos + Xpx*(X2-XX1);
          Vt_fin := Yoff*Ylen + Ypos + Ypx*(Y2-YY1);
        END; {With}
    END; {CalcPix}



  PROCEDURE CalcLevels;
    CONST
      MaxLevs  = 16;
      LevArray : ARRAY [0..MaxLevs-1] OF Single
                   =  (0, 0.05, 0.2, 0.4, 0.6, 0.7, 0.79, 0.86, 0.91, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 1);
    VAR
      Lev,Lev0       : Integer;
      LnRatio, Rat   : Single;
      MaxVal, MinVal : Single;

    PROCEDURE Calc1Lev;
      VAR
        ZZ, Z, Zs, A : Single;
        B, m, n      : Integer;
      BEGIN
        ZZ := Max ( ABS(MaxVal), ABS(MinVal) );
        Z  := Min ( ABS(MaxVal), ABS(MinVal) );
        Zs := ZZ / Sets.SnapNLev;
        IF (  (Zs >= Z)  OR  (MaxVal < 0)  OR  (MinVal > 0)  )
           THEN Sets.Snap1Lev := Exp (Ln(Sets.SnapNLev) / Sets.QtyLevels)
           ELSE BEGIN
                  {Assume Z > Zs}
                  A := ln (ZZ/Z) / ln (Z/Zs);     {A = m/n}
                  B := Sets.QtyLevels;            {B = 2n+1+m = 2n+1+An}
                  n := TRUNC ( (B-1) / (A+2) );
                  m := B - n - n - 1;
                  Sets.Snap1Lev := EXP ( ln(ZZ/Zs) / (m+n) );
                END;
      END; {Calc1Lev}

    BEGIN
      WITH Sets DO
        BEGIN
         {Set Max and Min values}
          IF (VectSet)
             THEN BEGIN  MaxVal := MaxV; MinVal := MinV;  END
             ELSE BEGIN
                    MaxVal := CurSnapRec.MaxVal;
                    MinVal := CurSnapRec.MinVal;
                    Rat := (QtyLevels-0.5)/QtyLevels;
                    IF (MaxVal >= 0)
                       THEN MaxVal := MaxVal*Rat
                       ELSE MaxVal := MaxVal/Rat;
                    IF (MINVal <= 0)
                       THEN MinVal := MinVal*Rat
                       ELSE MinVal := MinVal/Rat;
                  END;
          IF (SnapColAbs)
             THEN BEGIN
                    MaxVal := Max ( ABS(MaxVal), ABS(MinVal) );
                    MinVal := Max ( 0, MinVal);
                  END;
          IF (MaxVal = MinVal)
             THEN MaxVal := MinVal + 1;

          IF (SnapFill <> Log)
             THEN BEGIN {all non-log distributions}
                    Lev0 := 1;
                    FOR Lev := 1 TO QtyLevels DO
                      BEGIN
                        CASE SnapFill OF
                          Lin   : Level[Lev] := ((Lev-1)/(QtyLevels-1)) * (MaxVal-MinVal) + MinVal;
                          Quad  : Level[Lev] := SQR ((Lev-1)/(QtyLevels-1)) * (MaxVal-MinVal) + MinVal;
                          Fixed : Level[Lev] := LevArray [TRUNC ((Lev-1)/(QtyLevels-1)*(MaxLevs-1) )] * MaxVal;
                          DQuad : Level[Lev] := SQR (SQR ((Lev-1)/(QtyLevels-1)) ) * (MaxVal-MinVal) + MinVal;
                          Cosn  : Level[Lev] := 0.5 * SQR(SQR(1-COS (PI *  (Lev-1)/(QtyLevels-1) ) ) )
                                                    * (MaxVal-MinVal) + MinVal;
                        END; {Case}
{                        IF (Level[Lev] <= 0)  THEN Lev0 := Lev;}
                        IF (Lev > 1)
                           THEN IF (ABS(Level[Lev])<ABS(Level[Lev0]))
                                   THEN Lev0:=Lev;
                      END; {For}
                      Level [Lev0] := 0;
                    END {Non-log distribution}
             ELSE BEGIN
                    IF (SnapNLev > 0)
                       THEN Calc1Lev;
                    IF (MinVal=0)
                       THEN Lev0 := 1
                       ELSE IF (MaxVal=0)
                               THEN Lev0 := QtyLevels
                               ELSE Lev0 := TRUNC ( (QtyLevels - Ln( ABS(MaxVal/MinVal)) /Ln(Snap1Lev) ) /2) + 1;
                    IF ( (Lev0 > QtyLevels) OR (MaxVal < 0) )
                       THEN Lev0 := QtyLevels
                       ELSE IF ( (Lev0 <= 0) OR (MinVal > 0) )
                               THEN Lev0 := 1;

                    IF ( ABS (MinVal) > ABS (MaxVal) )
                       THEN BEGIN
                              Level[1] := MinVal;
                              FOR Lev := 2 TO (Lev0-1) DO
                                Level [Lev] := Level [Lev-1] / Snap1Lev;
                              IF (Lev0 < QtyLevels)
                                 THEN BEGIN
                                        Level [Lev0+1] := -Level [Lev0-1];
                                        FOR Lev := (Lev0+2) TO QtyLevels DO
                                            Level [Lev] := Level [Lev-1] * Snap1Lev;
                                      END;
                            END
                       ELSE BEGIN
                              Level [QtyLevels] := MaxVal;
                              FOR Lev := (QtyLevels-1) DOWNTO (Lev0 + 1) DO
                                Level [Lev] := Level [Lev+1] / Snap1Lev;
                              IF (Lev0 > 1)
                                 THEN BEGIN
                                        Level [Lev0-1] := -Level [Lev0+1];
                                        FOR Lev := (Lev0-2) DOWNTO 1 DO
                                            Level [Lev] := Level [Lev+1] * Snap1Lev;
                                      END;
                            END;
                    Level [Lev0] := 0;
                  END; {Log distribution}

          FOR Lev := 1 TO Sets.QtyLevels DO
              IF (Level[Lev] = 0)
                 THEN IF (SnapSet = ColMap)
                         THEN LevCol [Lev] := DsCol_Bgrnd
                         ELSE LevCol [Lev] := DsCol_Fgrnd2
                 ELSE LevCol [Lev] := ColOffset + ( (Sets.QtyCols * Lev) DIV (Sets.QtyLevels+1));

        END; {With SnapMap^ and Sets}
    END; {CalcLevels}


(*     FUNCTION GetLevel (Value : Single): Word;
       {Returns a level from 0 to (Sets.Qtylevels-1),
        based on the current setting for the level-setting method}
       {Returns CLOSEST level - OBSOLETE method}
     VAR
       H,L,C : Byte;
     BEGIN
       IF (Sets.SnapColAbs)
          THEN Value := ABS (Value);
       L := 1;
       H := Sets.QtyLevels;
       REPEAT
         C := (L+H) DIV 2;
         IF (Value < Level [C])
            THEN H := C
            ELSE L := C;
       UNTIL (L>=H-1);
       {check which level it is closest to}
       IF (H >= 2)
          THEN IF ( ABS (Value - Level[H-1]) < ABS (Value - Level[H]) )
                  THEN Dec(H);
       GetLevel := H;
     END; {GetLevel}  *)


   FUNCTION GetLevel (Value : Single): Word;
       {Returns a level from 0 to (Sets.Qtylevels-1),
        based on the current setting for the level-setting method}
       {Returns level, only if abs. greater}
     VAR
       H,L,C : Byte;
     BEGIN
       IF (Sets.SnapColAbs)
          THEN Value := ABS (Value);
       L := 1;
       H := Sets.QtyLevels;
       REPEAT
         C := (L+H) DIV 2;
         IF (Value < Level [C])
            THEN H := C
            ELSE L := C;
       UNTIL (L>=H-1);
       {check which level it is closest to}
       IF ( (Value >= 0) AND (Value<Level[H]) )
          THEN Dec(H);
       IF ( (Value<0) AND (Value<=Level[L]) )
          THEN H:=L;
       GetLevel := H;
     END; {GetLevel}


   PROCEDURE Ds_WriteLevel ( Pos  : Word;
                             Lev  : Single   );
     VAR
       Vpos, Hpos  : Word;
       LevMx       : Single;
       S           : String;
     BEGIN
       SetTextStyle (SmallFont, HorizDir, FontSz);
(*         IF (Sets.VGA256)
          THEN SetTextStyle (SmallFont, HorizDir, FontSz{2})
          ELSE SetTextStyle (SmallFont, HorizDir, FontSz{-1});*)
       SetTextJustify (CenterText, CenterText);
       LevMx := Max ( ABS(Level[Sets.QtyLevels]), ABS(Level[1]) );
       Vpos  := TRUNC ( LegWind.Sz[2] / 2 * (1 + Pos/(12+2) ) );
       Hpos  := LegWind.Sz[1] DIV 2;
       Ds_SetWindow (LegWind);
       MoveTo (Hpos, Vpos);
       S := Strng (Pos, $00);
       IF (Length(S) = 1)
          THEN Insert (' ', S, 1);
       OutText ( '['+ S + '] ' + Strng (Lev/LevMx,$F300) );
     END; {Ds_WriteLevel}


   PROCEDURE Ds_GraphContours;
     VAR
       ContQty, C, l  : Word;
       Dy, Dx         : Integer;
       xx, yy         : Integer;
       Xfin, Yfin     : Integer;
       Hz_st, Hz_blk, Hz_fin, Vt_st, Vt_blk, Vt_fin : LongInt;
       Vt_sgn         : Integer;
       HzOff, VtOff   : Single;
       Xscale, Yscale : Single;
       cfact          : Single;
       ContourPt      : ARRAY  [1..2] OF Point;
       BoxPt          : ARRAY  [0..4] OF RECORD
                                           Pt   : Point;
                                           Cval : Single;
                                         END;

     FUNCTION TestContour ( C1, C2 : Word;   Val3 : Single; PtNum : Word ) : Boolean;
       VAR
         Slope      : Single;
       BEGIN
         IF (   (  (BoxPt[C1].Cval >= Val3) AND (Val3 > BoxPt[C2].Cval)  ) OR
                (  (BoxPt[C2].Cval >= Val3) AND (Val3 > BoxPt[C1].Cval)  )   )
            THEN BEGIN
                   Slope := (Val3 - BoxPt[C1].Cval) / (BoxPt[C2].Cval - BoxPt[C1].Cval);
                   ContourPt [PtNum,X] := BoxPt[C1].Pt[X] + (BoxPt[C2].Pt[X] - BoxPt[C1].Pt[X]) * Slope;
                   ContourPt [PtNum,Y] := BoxPt[C1].Pt[Y] + (BoxPt[C2].Pt[Y] - BoxPt[C1].Pt[Y]) * Slope;
                   TestContour := True;
                 END
            ELSE TestContour := False;
       END; {TestContour}

     PROCEDURE DrawCont (P1, P2 : Point);
       VAR
         Px1, Px2 : Pixels;
       BEGIN
         Px1[1] := ROUND (HzOff + Xscale * P1[X]);
         Px1[2] := ROUND (VtOff + Yscale * P1[Y]);
         Px2[1] := ROUND (HzOff + Xscale * P2[X]);
         Px2[2] := ROUND (VtOff + Yscale * P2[Y]);
         Line (Px1[1], Px1[2], Px2[1], Px2[2]);
       END; {DrawCont}


     BEGIN {Ds_GraphContours}
       SetLineType (1);
       CalcLevels;
       CalcPix (Hz_st,Hz_blk,Hz_fin, Vt_st,Vt_blk,Vt_fin, Vt_sgn);
       WITH CurSnapRec DO
       WITH Scale2D DO
         BEGIN
           Xfin   := QuickX;  Yfin   := QuickY;
           Xscale := Hz_blk * (Xqty-1) / (QuickX-1) / Xlen;
           Yscale := Vt_blk * (Yqty-1) / (QuickY-1) / Ylen;
           HzOff  := Hz_st / Xlen;
           VtOff  := Vt_st / Ylen;
         END;

       IF (Sets.QtyLevels < MaxCont)
          THEN ContQty := Sets.QtyLevels
          ELSE ContQty := MaxCont;
       cfact := Sets.QtyLevels / (ContQty+1);
       FOR c := 1 TO ContQty DO
         IF (NOT(Keypressed)) THEN
            BEGIN
              l := TRUNC (c*cfact + 1);
              SetColor (LevCol[l]);
              FOR Dy := 1 TO Yfin-1 DO
                FOR Dx := 1 TO Xfin-1 DO
                  BEGIN
                    FOR xx := 0 TO 1 DO
                      FOR yy := 0 TO 1 DO
                        WITH BoxPt [2*xx+yy] DO
                          BEGIN
                            Pt[X] := (Dx-1+xx);
                            Pt[Y] := (Dy-1+yy);
                            Cval  := SnapShot^ [Dx+xx, Dy+yy];
                            IF (Sets.SnapColAbs)
                               THEN Cval := ABS (Cval);
                          END;
                    BoxPt[4].Pt[X] := 0; BoxPt[4].Pt[Y] := 0; BoxPt[4].Cval := 0;
                    FOR xx := 0 TO 3 DO
                      BEGIN
                        BoxPt[4].Pt[X] := BoxPt[4].Pt[X] + BoxPt[xx].Pt[X];
                        BoxPt[4].Pt[Y] := BoxPt[4].Pt[Y] + BoxPt[xx].Pt[Y];
                        BoxPt[4].Cval  := BoxPt[4].Cval  + BoxPt[xx].Cval;
                      END;
                    BoxPt[4].Pt[X] := BoxPt[4].Pt[X] * 0.25;
                    BoxPt[4].Pt[Y] := BoxPt[4].Pt[Y] * 0.25;
                    BoxPt[4].Cval  := BoxPt[4].Cval  * 0.25;

                    IF ( TestContour ( 0, 1, Level[l], 1)AND
                         TestContour ( 0, 4, Level[l], 2) )
                            THEN DrawCont (ContourPt [1], ContourPt [2]);
                    IF ( TestContour ( 0, 1, Level[l], 1)AND
                         TestContour ( 1, 4, Level[l], 2) )
                            THEN DrawCont (ContourPt [1], ContourPt [2]);
                    IF ( TestContour ( 0, 4, Level[l], 1)AND
                         TestContour ( 1, 4, Level[l], 2) )
                            THEN DrawCont (ContourPt [1], ContourPt [2]);

                    IF ( TestContour ( 1, 3, Level[l], 1)AND
                         TestContour ( 1, 4, Level[l], 2) )
                            THEN DrawCont (ContourPt [1], ContourPt [2]);
                    IF ( TestContour ( 1, 3, Level[l], 1)AND
                         TestContour ( 4, 3, Level[l], 2) )
                            THEN DrawCont (ContourPt [1], ContourPt [2]);
                    IF ( TestContour ( 1, 4, Level[l], 1)AND
                         TestContour ( 4, 3, Level[l], 2) )
                            THEN DrawCont (ContourPt [1], ContourPt [2]);

                    IF ( TestContour ( 0, 2, Level[l], 1)AND
                         TestContour ( 0, 4, Level[l], 2) )
                            THEN DrawCont (ContourPt [1], ContourPt [2]);
                    IF ( TestContour ( 0, 2, Level[l], 1)AND
                         TestContour ( 2, 4, Level[l], 2) )
                            THEN DrawCont (ContourPt [1], ContourPt [2]);
                    IF ( TestContour ( 0, 4, Level[l], 1)AND
                         TestContour ( 2, 4, Level[l], 2) )
                            THEN DrawCont (ContourPt [1], ContourPt [2]);

                    IF ( TestContour ( 2, 3, Level[l], 1)AND
                         TestContour ( 2, 4, Level[l], 2) )
                            THEN DrawCont (ContourPt [1], ContourPt [2]);
                    IF ( TestContour ( 2, 3, Level[l], 1)AND
                         TestContour ( 4, 3, Level[l], 2) )
                            THEN DrawCont (ContourPt [1], ContourPt [2]);
                    IF ( TestContour ( 2, 4, Level[l], 1)AND
                         TestContour ( 4, 3, Level[l], 2) )
                            THEN DrawCont (ContourPt [1], ContourPt [2]);
                  END; {For Dx, Dy}
{                WriteLevel (c, Level[l]);}

            END; {Loop for each contour}
     END; {Ds_GraphContours}


  PROCEDURE Ds_LineGr;
    VAR
      Hz_st, Hz_blk, Hz_fin, Vt_st, Vt_blk, Vt_fin : LongInt;
      Vt_sgn             : Integer;
      xPix, yPix, ix, iy : Integer;
      Hz, Vt             : LongInt;
      Col                : Word;
      Lev                : Integer;
      Mx, Ang            : Single;
      CosLev, SinLev     : Integer;
    BEGIN {Ds_LineGr}
      SetLineType (1);
      CalcLevels;
      CalcPix (Hz_st,Hz_blk,Hz_fin, Vt_st,Vt_blk,Vt_fin, Vt_sgn);
      SetColor (DsCol_Fgrnd);
      WITH CurSnapRec DO
       WITH Scale2D DO
        BEGIN
          Mx := Max ( ABS(MaxVal), ABS(MinVal) );
          Hz_blk  := (Hz_blk*(Xqty-1)) DIV (QuickX-1);
          Vt_blk  := (Vt_blk*(Yqty-1)) DIV (QuickY-1);
          IF (Sets.VectSet)
             THEN Ds_VectSc := (Hpix/ScrnLen) * Sets.VectSc
             ELSE Ds_VectSc := Hz_blk / Xlen * Vzoom / Mx / LineGzoom;

          Hz := Hz_st;
          IF (Sets.Dirn IN [Vert, Both])
             THEN FOR ix := 1 TO QuickX DO
                    IF (NOT(Keypressed)) THEN
                       BEGIN
                         xPix := Hz DIV Xlen - 1;
                         Vt := Vt_st;
                         FOR iy := 1 TO QuickY DO
                           BEGIN
                             yPix := Vt DIV Ylen - Vt_sgn;
                             IF (Sets.LineCol)
                                THEN BEGIN
                                       Col := LevCol  [GetLevel(SnapShot^[ix,iy])];
                                       SetColor (Col);
                                     END;
                             Lev := ROUND (SnapShot^[ix,iy] * Ds_VectSc);
                             IF (Sets.Dirn = Vert)
                                THEN BEGIN  SinLev := 0; CosLev := Lev;  END
                                ELSE IF (Ax1 = 't')
                                        THEN BEGIN  SinLev := Lev; CosLev := Lev;  END
                                        ELSE BEGIN
                                               ang := Arctan0 ( ABS(QuickX*0.5 - ix), ABS(QuickY*0.5 - iy) );
                                               SinLev := ROUND (Lev * SIN(Ang) );
                                               CosLev := ROUND (Lev * COS(Ang) );
                                             END;
                             IF (iy = 1)
                                THEN MoveTo (xPix+CosLev, yPix-SinLev)
                                ELSE LineTo (xPix+CosLev, yPix-SinLev);
                             Vt := Vt + Vt_blk;
                           END; {For iy}
                         Hz := Hz + Hz_blk;
                       END; {for, if}

          Vt := Vt_st;
          IF (Sets.Dirn IN [Horiz, Both])
             THEN FOR iy := 1 TO QuickY DO
                    IF (NOT(Keypressed)) THEN
                       BEGIN
                         yPix := Vt DIV Ylen - Vt_sgn;
                         Hz := Hz_st;
                         FOR ix := 1 TO QuickX DO
                           BEGIN
                             xPix := Hz DIV Xlen - 1;
                             IF (Sets.LineCol)
                                THEN BEGIN
                                       Col := LevCol  [GetLevel ( SnapShot^[ix,iy] )];
                                       SetColor (Col);
                                     END;
                             Lev := ROUND (SnapShot^[ix,iy] * Ds_VectSc);
                             IF (Sets.Dirn = Horiz)
                                THEN BEGIN  SinLev := Lev; CosLev := 0;  END
                                ELSE IF (Ax1 = 't')
                                        THEN BEGIN  SinLev := Lev; CosLev := Lev;  END
                                        ELSE BEGIN
                                               ang := Arctan0 ( ABS(QuickX*0.5 - ix), ABS(QuickY*0.5 - iy) );
                                               SinLev := ROUND (Lev * SIN(Ang) );
                                               CosLev := ROUND (Lev * COS(Ang) );
                                             END;
                             IF (ix = 1)
                                THEN MoveTo (xPix+CosLev, yPix-SinLev)
                                ELSE LineTo (xPix+CosLev, yPix-SinLev);
                             Hz := Hz + Hz_blk;
                           END; {For iy}
                         Vt := Vt + Vt_blk;
                       END; {for, if}
        END; {With}
    END; {Ds_LineGr}


  PROCEDURE QuickSnap;
      {Reads snap values into fixed size array - max 101x101}
      {Truncates if Xqty or Yqty is too large}
    VAR
      Hz_st, Hz_blk, Hz_fin, Vt_st, Vt_blk, Vt_fin : LongInt;
      Vt_sgn             : Integer;
      xPix, yPix         : Pixels;
      ix, iy             : Integer;
      Hz, Vt             : LongInt;
      Col, Fill          : Word;
    BEGIN {QuickSnap}
      Fill := SolidFill;
      CalcLevels;
      CalcPix (Hz_st,Hz_blk,Hz_fin, Vt_st,Vt_blk,Vt_fin, Vt_sgn);
      WITH CurSnapRec DO
       WITH Scale2D DO
        BEGIN
          Hz_blk  := (Hz_blk*(Xqty-1)) DIV (QuickX-1);
          Vt_blk  := (Vt_blk*(Yqty-1)) DIV (QuickY-1);
          xPix[1] := Hz_st DIV Xlen;   Hz := Hz_st + Hz_blk DIV 2;
          FOR ix := 1 TO QuickX DO
            IF (NOT(Keypressed)) THEN
               BEGIN
                 IF (ix=QuickX)
                    THEN xPix[2] := Hz_fin DIV Xlen
                    ELSE xPix[2] := Hz DIV Xlen - 1;
                 yPix[1] := Vt_st DIV Ylen;  Vt := Vt_st + Vt_blk DIV 2;
                 FOR iy := 1 TO QuickY DO
                   BEGIN
                     Col := LevCol  [GetLevel ( SnapShot^[ix,iy] )];
                     SetFillStyle (Fill, Col);
                     IF (iy=QuickY)
                        THEN yPix[2] := Vt_fin DIV Ylen
                        ELSE yPix[2] := Vt DIV Ylen - Vt_sgn;
                     IF (Col <> Black)
                        THEN Bar (xPix[1], yPix[1], xPix[2], yPix[2]);
                     yPix[1] := yPix[2] + Vt_sgn;
                     Vt := Vt + Vt_blk;
                   END;
                 xPix[1] := xPix[2] + 1;
                 Hz := Hz + Hz_blk;
               END;
        END; {With}
    END; {QuickSnap}


  PROCEDURE InterpSnap;
      {Plots interpolated blocks - UNFORTUNATELY uses PutPixel}
    VAR
      Hz_st, Hz_blk, Hz_fin, Vt_st, Vt_blk, Vt_fin : LongInt;
      Vt_sgn             : Integer;
      xPix, yPix         : Pixels;
      xp, yp             : Integer;
      ix, iy             : Integer;
      Hz, Vt             : LongInt;
      Col, Fill          : Word;
      i,j, II, JJ        : Word;
      IJ_i               : Single;
      Lev                : Single;
      C                  : ARRAY [0..1,0..1] OF Single;
    BEGIN {InterpSnap}
      Fill := SolidFill;
      CalcLevels;
      CalcPix (Hz_st,Hz_blk,Hz_fin, Vt_st,Vt_blk,Vt_fin, Vt_sgn);
      WITH CurSnapRec DO
       WITH Scale2D DO
        BEGIN
          Hz_blk  := (Hz_blk*(Xqty-1)) DIV (QuickX-1);
          Vt_blk  := (Vt_blk*(Yqty-1)) DIV (QuickY-1);
          xPix[1] := Hz_st DIV Xlen;   Hz := Hz_st + Hz_blk;
          FOR ix := 1 TO (QuickX-1) DO
            IF (NOT(Keypressed)) THEN
               BEGIN
                 IF (ix=QuickX-1)
                    THEN xPix[2] := Hz_fin DIV Xlen
                    ELSE xPix[2] := Hz DIV Xlen - 1;
                 yPix[1] := Vt_st DIV Ylen;  Vt := Vt_st + Vt_blk;
                 FOR iy := 1 TO (QuickY-1) DO
                   BEGIN
                     IF (iy=QuickY-1)
                        THEN yPix[2] := Vt_fin DIV Ylen
                        ELSE yPix[2] := Vt DIV Ylen - Vt_sgn;;
                     C[0,0] := ( SnapShot^[ix  ,iy  ] );
                     C[1,0] := ( SnapShot^[ix+1,iy  ] );
                     C[0,1] := ( SnapShot^[ix  ,iy+1] );
                     C[1,1] := ( SnapShot^[ix+1,iy+1] );
                     II := ABS(xPix[2]-xPix[1])+1;
                     JJ := ABS(yPix[2]-yPix[1])+1;
                     IJ_i := 1.0/(II*JJ);    {Take inverse for speed}
                     yp := yPix[1];
                     FOR j := 1 TO JJ DO
                       BEGIN
                         xp := xPix[1];
                         FOR i := 1 TO II DO
                           BEGIN
                             Lev := (  (C[0,0]*(JJ-j) + C[0,1]*j) * (II-i)
                                     + (C[1,0]*(JJ-j) + C[1,1]*j) * i  ) * IJ_i;
                             Col := LevCol [GetLevel (Lev)];
                             IF (Col <> Black)  THEN PutPixel (xp, yp, Col);
                             xp := xp+1;
                           END;  {For i}
                         yp := yp+Vt_sgn;
                       END;  {For j}
                     yPix[1] := yPix[2] + Vt_sgn;
                     Vt := Vt + Vt_blk;
                   END;
                 xPix[1] := xPix[2] + 1;
                 Hz := Hz + Hz_blk;
               END;
        END; {With}
    END; {InterpSnap}


  PROCEDURE FullSnap;
    VAR
      Hz_st, Hz_blk, Hz_fin, Vt_st, Vt_blk, Vt_fin : LongInt;
      Vt_sgn             : Integer;
      xPix, yPix         : Pixels;
      ix, iy             : Integer;
      Hz, Vt             : LongInt;
      Col, Fill          : Word;
      SnapF              : File;
      SnapVal            : ARRAY [0..1601] OF Single;
    BEGIN {FullSnap}
      Fill := SolidFill;
      CalcLevels;
      CalcPix (Hz_st,Hz_blk,Hz_fin, Vt_st,Vt_blk,Vt_fin, Vt_sgn);
      ASSIGN (SnapF,     (Sets.Drive + ':' + Sets.Dir + '\' + Sets.FName + '.SNP') );
      RESET  (SnapF, 4);
      WITH SnapMap^ [CurSnap] DO    {Note, info required which is not stored in CurSnapRec}
       WITH Scale2D DO
        BEGIN
          Seek (SnapF, Offset);
          xPix[1] := Hz_st DIV Xlen;
          Hz := Hz_st + Hz_blk DIV 2;  {Half block so doesn't cross boundary}
          FOR ix := 1 TO (Xqty) DO
            IF (NOT(Keypressed)) THEN
               BEGIN
                 IF (ix=Xqty)
                    THEN xPix[2] := Hz_fin DIV Xlen
                    ELSE xPix[2] := Hz DIV Xlen - 1;
                 BlockRead (SnapF, SnapVal, Yqty+2); {1 extra 4-byte variable on either side}
                 yPix[1] := Vt_st DIV Ylen;  Vt := Vt_st + Vt_blk DIV 2;
                 FOR iy := 1 TO Yqty DO
                   BEGIN
                     Col := LevCol  [GetLevel (SnapVal[iy])];
                     SetFillStyle (Fill, Col);
                     IF (iy=Yqty)
                        THEN yPix[2] := Vt_fin DIV Ylen
                        ELSE yPix[2] := Vt DIV Ylen - Vt_sgn;
                     IF (Col <> Black)
                        THEN Bar (xPix[1], yPix[1], xPix[2], yPix[2]);
                     yPix[1] := yPix[2] + Vt_sgn;
                     Vt := Vt + Vt_blk;
                   END;
                 xPix[1] := xPix[2] + 1;
                 Hz := Hz + Hz_blk;
               END;
        END; {With}
      CLOSE (SnapF);
    END; {FullSnap}


  PROCEDURE Ds_ListLevels;
    VAR
      Lev, l       : Word;
      LevMx        : Single;
      S            : String;
      Mid          : Word;
      LevList     : Word;
      Vstep        : Single;
    BEGIN {Ds_ListLevels}
      Ds_SetWindow (LegWind);
      Vstep := LegWind.Sz[2] * 0.5;
      Mid   := LegWind.Sz[1] DIV 2;
{      IF (Sets.VGA256)
         THEN SetTextStyle (SmallFont, HorizDir, FontSz-2)
         ELSE SetTextStyle (SmallFont, HorizDir, FontSz-1);}
      SetTextStyle (SmallFont, HorizDir, FontSz);
      LevList := LevListMx;
(*      IF (Sets.VGA256)
         THEN LevList := 13{8}
         ELSE LevList := 13;*)
      WITH Sets DO
        LevList := (QtyLevels-1) DIV ( ((QtyLevels-1) DIV LevListMx) + 1) + 1;
(*        IF ( (Sets.QtyLevels MOD LevList) <> 0 )
           THEN LevList := Sets.QtyLevels DIV ((Sets.QtyLevels DIV (LevList-1))) + 1;*)
      WITH CurSnapRec DO
      WITH LegWind DO
        BEGIN
          SetColor (DsCol_Fgrnd);
          SetTextStyle (SmallFont, HorizDir, FontSz);
          SetTextJustify (CenterText, CenterText);
          MoveTo (Mid, TRUNC(Vstep * 0.100)  );
          OutText ('SNAPSHOT ' + Strng (Snp,0) );

          SetTextStyle (SmallFont, HorizDir, FontSz-1);
          MoveTo ( Mid, TRUNC(Vstep*0.190) );
          OutText (' File = '+Sets.Fname);
          MoveTo ( Mid, TRUNC(Vstep*0.250) );
          OutText (' ('+Sets.Drive+':'+Sets.Dir+'\)');

          SetTextJustify (LeftText, CenterText);
          SetTextStyle (SmallFont, HorizDir, FontSz);
          MoveTo ( Hinset, TRUNC(Vstep * 0.375)  );
          OutText (' Var  = ' + VarName(SnapVar) + ' '+VFuncName[SFunc] );
          MoveTo ( Hinset, TRUNC(Vstep * 0.450)  );
          OutText (' Time = ' + Strng (Time,$F400) );
          MoveTo ( Hinset, TRUNC(Vstep * 0.525)  );
          OutText (' Max  = ' + Strng (MaxVal,$F400) );
          MoveTo ( Hinset, TRUNC(Vstep * 0.600)  );
          OutText (' Min  = ' + Strng (MinVal,$F400) );
          IF (Sets.SnapSet = MLine)
             THEN BEGIN
                    SetTextJustify (CenterText, CenterText);
                    MoveTo (Sz[1] DIV 2, TRUNC(Vstep * 0.750)  );
                    OutText ('Vector Scale');
                    SetTextStyle (SmallFont, HorizDir, FontSz-1);
                    MoveTo (Sz[1] DIV 2, TRUNC(Vstep * 0.825)  );
                    OutText ('1CM : ' + Strng ( (Hpix/ScrnLen) / Ds_VectSc, $F300 )  );
                  END;
          IF (Sets.MultSnap AND MultNolink)
             THEN BEGIN
                    SetTextJustify (CenterText, CenterText);
                    MoveTo (Sz[1] DIV 2, TRUNC(Vstep * 0.675)  );
                    OutText ('NOT LINKED');
                  END;

          SetTextStyle (SmallFont, HorizDir, FontSz);
          IF ( (Sets.SnapSet <> MLine) OR (Sets.LineCol) )
             THEN BEGIN
                    SetTextJustify (CenterText, CenterText);
                    LevMx   := Max ( ABS(Level[Sets.QtyLevels]), ABS(Level[1]) );
                    MoveTo (Mid, TRUNC(Vstep * 0.950)  );
                    OutText ('Colour Values');
                    MoveTo (Mid, TRUNC(Vstep * 1.025)  );
                    SetTextStyle (SmallFont, HorizDir, FontSz-1);
                    OutText ('(Distribution = '+ SnapFillNames[Sets.SnapFill]+')');
                    SetTextStyle (SmallFont, HorizDir, FontSz);
{                    IF (Sets.VidMode=std16)
                       THEN SetFillStyle(solidFill, White)
                       ELSE SetFillStyle(solidFill, LightGray);
                    Bar (TRUNC(Mid*0.2), TRUNC(Vstep*1.05), TRUNC(Mid*1.8), TRUNC(Vstep*1.95));}
                    IF (Sets.SnapSet IN [ColMap, Mline, ContMap])
                       THEN FOR l := 0 TO (LevList-1) DO
                             BEGIN
                               Lev := 1 + ROUND (l * (Sets.QtyLevels-1) / (LevList-1) );
                               IF (LevCol [Lev] = DsCol_Bgrnd)
                                  THEN BEGIN
                                         IF (Sets.VidMode=std16)
                                            THEN SetFillStyle(solidFill, White)
                                            ELSE SetFillStyle(solidFill, DarkGray);
                                         Bar ( TRUNC(Mid*0.2), TRUNC(Vstep * (1.1 + (LevList-l-0.80)/LevList/1.25) ),
                                               TRUNC(Mid*1.8), TRUNC(Vstep * (1.1 + (LevList-l+0.10)/LevList/1.25) )   );
                                       END;
                               MoveTo ( Mid, TRUNC( Vstep * (1.1 + (LevList-l-0.5) / LevList / 1.25)  )  );
                               SetColor (LevCol [Lev]);
                               S := Strng (Lev-1, $00);
                               IF (Length(S) = 1)
                                  THEN Insert (' ', S, 1);
                               OutText ( '['+ S + '] ' + Strng(Level[Lev],$F400) );
                             END;
                  END; {Colour Listing}
        END; {With}
      ResetViewPort;
    END; {Ds_ListLevels}


  PROCEDURE Ds_GraphSnap;
    BEGIN
      Ds_SetWindow(SnapWind);
      WITH Sets DO
        CASE SnapSet OF
           ColMap : IF (Interp)
                       THEN InterpSnap
                       ELSE IF (Quick)
                               THEN QuickSnap
                               ELSE FullSnap;
           ContMap : Ds_GraphContours;
           MLine   : Ds_LineGr;
        END; {Case}
      ResetViewPort;
      IF (Sets.ShowTitles) THEN Ds_Titles;
    END; {Ds_GraphSnap}


  {Temporary addition}
    PROCEDURE DI_CvertAscii2 (MxMin: Integer);
      VAR
        ix, iy    : Integer;
        AsciiFile : Text;
        Lmax      : ARRAY [1..1600] OF Single;

      PROCEDURE MaxLine (MxMin:Integer);
        VAR
          ix, iy             : Integer;
          SnapF              : File;
          SnapVal            : ARRAY [0..1601] OF Single;
        BEGIN {MaxLine}
          ASSIGN (SnapF,     (Sets.Drive + ':' + Sets.Dir + '\' + Sets.FName + '.SNP') );
          RESET  (SnapF, 4);
          WITH SnapMap^ [CurSnap] DO
    {       WITH Scale2D DO}
            BEGIN
              FOR iy := Yqty DOWNTO 1 DO
                CASE MxMin OF
                  0 : Lmax[iy] := -LargeSingle;  {Max reqd}
                  1 : Lmax[iy] := LargeSingle;   {Min reqd}
                  2 : Lmax[iy] := 0;             {ABS Max reqd}
                  ELSE Lmax[iy] := 0;            {ABS Max reqd}
                END; {Case}
              Seek (SnapF, Offset);
              FOR ix := 1 TO (Xqty) DO
                BEGIN
                  BlockRead (SnapF, SnapVal, Yqty+2); {1 extra 4-byte variable on either side}
                  FOR iy := Yqty DOWNTO 1 DO
                    CASE MxMin OF
                      0 : IF (SnapVal[iy]>Lmax[iy]) THEN Lmax[iy]:=SnapVal[iy];
                      1 : IF (SnapVal[iy]<Lmax[iy]) THEN Lmax[iy]:=SnapVal[iy];
                      2 : IF (ABS(SnapVal[iy])>Lmax[iy]) THEN Lmax[iy]:=ABS(SnapVal[iy]);
                      ELSE  IF (ABS(SnapVal[iy])>Lmax[iy]) THEN Lmax[iy]:=ABS(SnapVal[iy]);
                    END; {Case}
                END; {For}
            END; {With}
          CLOSE (SnapF);
        END; {MaxLine}

      BEGIN {DI_CvertAscii2}
        MaxLine(MxMin);
        ASSIGN (AsciiFile, Sets.Drive + ':' + Sets.Dir + '\' + Sets.FName + '.ASC' );
        {$I-} {Turn off IO error checking}
          APPEND (AsciiFile);
          IF (IOResult <> 0)
             THEN REWRITE (AsciiFile);
        {$I+} {Turn on IO error checking}
        WITH SnapMap^ [CurSnap] DO
          FOR iy := 1 TO Yqty DO
            writeln (AsciiFile, ({j1+}iy-1)*dy:14, '    ', Lmax[iy]:14 );
        CLOSE (AsciiFile);
      END; {DI_CvertAscii2}

