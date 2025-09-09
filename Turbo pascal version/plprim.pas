{RESOURCE: Plotter Resource}

 UNIT PlPrim;

  {Produces plotter output using RD-GL/ HPGL graphic language constructs}
  {Contains constants, data structures & operators, used to access the plotter}


{* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * }

  INTERFACE

     USES
       Funct, Graph, Setts, Pl_BGI, Dsprim;

     CONST
       Sratio = 1.05;

       PlPen_Bgrnd  = 0;
       PlPen_Fgrnd  = 1;
       PlPen_Fgrnd2 = 7;

       PlPen_Axes   = PlPen_Fgrnd;
       PlPen_Val    = PlPen_Fgrnd;
       PlPen_Scal   = PlPen_Fgrnd;
       PlPen_List   = PlPen_Fgrnd;
       PlPen_Seism  = PlPen_Fgrnd+1;

       PlPen_Border  = PlPen_Fgrnd;
       PlPen_CStress = 2;
       PlPen_TStress = 3;
       PlPen_Vect    = 2;
       PlPen_Arrow   = 3;
       PlPen_Geom    = PlPen_Fgrnd;
       PlPen_Titles  = PlPen_Fgrnd;

     TYPE
       Pl_Window  = Pl_Win;

     VAR
       Pl_SeismWind         : ARRAY [1..MaxSeism] OF
                                    ARRAY [1..3]  OF Pl_Window;
       Pl_SnapWind, Pl_XtrWind, Pl_LegWind2,
         Pl_LegWind, Pl_GraphWind         : Pl_Window;
       Pl_HzTWind, Pl_VtTWind, Pl_HdTWind : Pl_Window;

       Pl_VectSc            : Single;
       Pl_Hscale, Pl_Vscale : Integer;
       Pl_Xpx, Pl_Ypx       : LongInt;
       Pl_Xoff, Pl_Yoff     : Integer;
       PgLenHz, PgLenVt     : Single;


{* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * }


   {Extensions to the BGI}

     PROCEDURE Pl_SetLineType (Lnum : Integer);

     PROCEDURE Pl_ResetLineStyle;

     PROCEDURE Pl_ResetTextStyle;

     PROCEDURE Pl_SetText ( Hoffset, Voffset : Integer;
                            Hjust, Vjust, Font, FontMagn : Word );


   {Initialization of Graphics System}

     PROCEDURE Pl_Init;


   {Windowing Routines}

     PROCEDURE Pl_SetWindow (w: Pl_Window);

     PROCEDURE Pl_DrawWindow (w: Pl_Window);

     PROCEDURE Pl_InitWindows;

     PROCEDURE Pl_CalcSeismWind;


  {Macro Routines for graphing}

     PROCEDURE Pl_Draw_CM;

     PROCEDURE Pl_DrawAng (Hz_St, Vt_St, Len, Ang: Single);

     PROCEDURE Pl_DrawArrow (Hz_tip, Vt_tip, Hz_len, Vt_len: Single);

     PROCEDURE Pl_CentreBar (Hz_st, Vt_st, Len, Ang: Single);

     PROCEDURE Pl_DrawCrack (Hz_st, Vt_st, Hz_end, Vt_end: Single);



{* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *}

  IMPLEMENTATION


  {Extensions to the BGI}

     PROCEDURE Pl_SetLineType (Lnum : Integer);
       BEGIN
         CASE Lnum OF
            1 : Writeln (Pl, Pl_LineType,';');
            2 : Writeln (Pl, Pl_LineType, 2, Sep, '1.0');
            3 : Writeln (Pl, Pl_LineType, 1, Sep, '0.25');
{              2 : Writeln (Pl, Pl_LineType, 2, Sep, '0.5');
            3 : Writeln (Pl, Pl_LineType, 1, Sep, '0.25');}
            4 : Writeln (Pl, Pl_LineType, 6, Sep, '2.5');
            5 : Writeln (Pl, Pl_LineType, 4, Sep, '2.5');
            6 : Writeln (Pl, Pl_LineType, 2, Sep, '1.5');
            ELSE Writeln (Pl, Pl_LineType,';');
         END; {Case}
       END; {Pl_SetLineType}

     PROCEDURE Pl_ResetLineStyle;
       BEGIN
         Writeln (Pl, Pl_LineType,';');
         Writeln (Pl, Pl_PenThick,'0.5');
       END; {Pl_ReSetLineStyle}


     PROCEDURE Pl_ResetTextStyle;
       BEGIN
         Pl_TxtDir := HorizDir;
         Pl_SetTextStyle (DefaultFont, Pl_TxtDir, 4);
       END; {Pl_ResetTextStyle}

     PROCEDURE Pl_SetText ( Hoffset, Voffset : Integer;
                            Hjust, Vjust, Font, FontMagn : Word );
       BEGIN
         Pl_SetTextStyle (Font, HorizDir, FontMagn);
         Pl_MoveRel ( Hoffset, -Voffset);
         Pl_SetTextJustify (Hjust, Vjust);
       END; {Pl_SetText}


  {Graphics Initialization}

     PROCEDURE Pl_Init;
       BEGIN
         Pl_InitWindows;
       END; {Pl_Init}


  {Windowing Routines}

     PROCEDURE Pl_SetWindow (w: Pl_Window);
       BEGIN
         WITH w DO
           Pl_SetViewPort (St[1], St[2], Fin[1], Fin[2]);
       END; {Pl_SetWindow}

     PROCEDURE Pl_DrawWindow (w: Pl_Window);
       BEGIN
         Pl_ResetViewPort;
         WITH w DO
           Pl_Rectangle (St[1]-1, St[2]-1, Fin[1]+1, Fin[2]+1);
         Pl_SetWindow (w);
       END; {Pl_DrawWindow}


     PROCEDURE Pl_InitWindows;
       VAR
         Vsz, Voff, Hsz, Hoff : LongInt;
       BEGIN
         Vsz  := SheetSizes[2]-100-2;
         Hsz  := Vsz;
         Voff := Vsz DIV 20;
         Hoff := Hsz DIV 20;

         WITH Pl_GraphWind DO
           BEGIN
             St[1]  := 100;  Sz[1]  := Hsz+2;  Fin[1] := St[1]+Sz[1]-1;
             St[2]  := 100;  Sz[2]  := Vsz+2;  Fin[2] := St[2]+Sz[2]-1;
           END; {With Pl_GraphWind}
         WITH Pl_XtrWind DO
           BEGIN
             St[1]  := Pl_GraphWind.Fin[1]+1;   Sz[1]  := (SheetSizes[1]-100-2)-St[1]+1;  Fin[1] := (SheetSizes[1]-100-2);
             Fin[2] := Pl_GraphWind.Fin[2];     Sz[2] := Vsz DIV 5;                       St[2] := Fin[2]-Sz[2]+1;
           END; {With Pl_XtrWind.}
         WITH Pl_LegWind DO
           BEGIN
             St[1]  := Pl_XtrWind.St[1];      Sz[1]  := Pl_XtrWind.Sz[1];      Fin[1]:=Pl_XtrWind.Fin[1];
             St[2]  := Pl_GraphWind.St[2];    Fin[2] := Pl_XtrWind.St[2] - 2;  Sz[2] := Fin[2]-St[2]+1;
           END; {With Pl_LegWind.}

         WITH Pl_HdTWind DO
           BEGIN
             St[1]  := Pl_GraphWind.St[1]+Hoff+1;   Sz[1]  := Hsz-Hoff;          Fin[1] := St[1]+Sz[1]-1;
             St[2]  := Pl_GraphWind.St[2]+Vsz-Voff; Sz[2]  := Voff;              Fin[2] := St[2] + Sz[2] - 1;
           END; {With Pl_HdTWind}
         WITH Pl_HzTWind DO
           BEGIN
             St[1]  := Pl_HdTWind.St[1];            Sz[1]  := Pl_HdTWind.Sz[1];  Fin[1] := Pl_HdTWind.Fin[1];
             St[2]  := Pl_GraphWind.St[2]+1;        Sz[2]  := Voff;              Fin[2] := St[2] + Sz[2] - 1;
           END; {With Pl_HzTWind}
         WITH Pl_VtTWind DO
           BEGIN
             St[1]  := Pl_GraphWind.St[1]+1;        Sz[1]  := Hoff;              Fin[1] := St[1]+Sz[1]-1;
             St[2]  := Pl_GraphWind.St[2]+Voff+1;   Sz[2]  := Vsz-Voff-Voff;     Fin[2] := St[2]+Sz[2]-1;
           END; {With Pl_VtTWind}

         WITH Pl_SnapWind DO
           BEGIN
             St[1]  := Pl_VtTWind.Fin[1]+1;    Sz[1] := Hsz-Hoff;         Fin[1] := St[1]+Sz[1]-1;
             St[2]  := Pl_HzTWind.Fin[2]+1;    Sz[2] := Vsz-Voff-Voff;    Fin[2] := St[2]+Sz[2]-1;
             Pl_Vscale  := Pl_SnapWind.Sz[2]-1;
             Pl_Hscale  := Pl_SnapWind.Sz[1]-1;
           END; {With Pl_SnapWind}

         PgLenHz := Pl_Hscale/Pl_1CM;
         PgLenVt := Pl_Vscale/Pl_1CM;
       END;  {Pl_InitWindows}


     PROCEDURE Pl_CalcSeismWind;
       CONST
         HzRat : ARRAY [1..6] OF Single = (0.0, 0.10, 0.11, 0.850, 0.86, 1.0);
         VtRat : ARRAY [1..2] OF Single = (0.04, 0.96);
       VAR
         i,j   : Word;
         Hz    : ARRAY [1..6] OF Word;
         Vt    : ARRAY [1..2] OF Word;
         VSize : Single;
         Voff  : LongInt;
       BEGIN
         WITH Pl_SnapWind DO
           BEGIN
             FOR i := 1 TO 6 DO
               Hz[i] := St[1] + TRUNC (Sz[1]*HzRat[i]);
             VSize := Sz[2] / QtySWind;
             Voff  := St[2];
           END; {With}
         FOR i := 1 TO QtySWind DO
           BEGIN
             Vt[1] := TRUNC ( (QtySWind-i)*Vsize + VtRat[1]*Vsize) + Voff;
             Vt[2] := TRUNC ( (QtySWind-i)*Vsize + VtRat[2]*Vsize) + Voff;
             FOR j := 1 TO 3 DO
               WITH Pl_SeismWind[i,j] DO
                 BEGIN
                   St[1]  := Hz[j*2-1];  Fin[1] := Hz[j*2];  Sz[1]  := Fin[1] - St[1] + 1;
                   St[2]  := Vt[1];      Fin[2] := Vt[2];    Sz[2]  := Fin[2] - St[2] + 1;
                 END; {With}
           END; {For}
         IF (SingleWind)
            THEN FOR i := 2 TO QtySeism DO
                   Pl_SeismWind[i] := Pl_SeismWind[1];
(*         Pl_SelectColour (PlPen_Axes);
         Pl_DrawWindow (Pl_XtrWind);
         Pl_DrawWindow (Pl_LegWind);*)
(*         FOR i := 1 TO QtySWind DO
           FOR j := 1 TO 3 DO
             Pl_DrawWindow (Pl_SeismWind[i,j]); *)
       END;  {Pl_CalcSeismWind}



  {Routines for Scaling: in Plot unit}


  {Macro Routines for graphing}


{       PROCEDURE Pl_DrawLine (P1, P2 : Point);
         VAR
           Px1, Px2 : Pl_Point;
         BEGIN
           Px1[1] := IROUND (Pl_PScale[X] * P1[X]);
           Px1[2] := IROUND (Pl_PScale[Y] * P1[Y]);
           Px2[1] := IROUND (Pl_PScale[X] * P2[X]);
           Px2[2] := IROUND (Pl_PScale[Y] * P2[Y]);
           Pl_Line (Px1[1], Px1[2], Px2[1], Px2[2]);
         END; {Pl_DrawLine}


       PROCEDURE Pl_Draw_CM;
         CONST
           Len  = Pl_1cm;   {gives 1 cm on the plotter}
         BEGIN
           Pl_MoveRel (-Len DIV 2, 0);
           Pl_LineRel (Len, 0);
         END; {Pl_Draw_CM}


       PROCEDURE Pl_DrawAng (Hz_St, Vt_St, Len, Ang: Single);
         BEGIN
           Pl_Line ( IROUND(Hz_St), IROUND(Vt_St), IROUND (Hz_St + Len*COS(Ang)),
                                   IROUND (Vt_St + Len*SIN(Ang)) );
         END; {Pl_DrawAng}


       PROCEDURE Pl_DrawArrow (Hz_tip, Vt_tip, Hz_len, Vt_len: Single);
         CONST
           ArrowAng     = 0.4; {10 Degrees}
           CosArrow     = 0.92106099400;             {COS (ArrowAng)}
           SinArrow     = 0.38941834231;             {SIN (ArrowAng)}
           MinArrowLen  = ROUND (0.5 * Pl_1cm / 10); {0.5 mm on plotter}
         VAR
           Ang, Arrowlen  : Single;
           CosAng, SinAng : Single;
         BEGIN
           ArrowLen := SQRT ( SQR(Hz_len) + SQR(Vt_len) )/6;
           Ang      := Arctan0 (Vt_len, Hz_len);
           CosAng   := COS (Ang);
           CosAng := COS (Ang);
           SinAng := SIN (Ang);
           IF (ArrowLen < MinArrowLen)
              THEN ArrowLen := MinArrowLen;
           Pl_Line ( IROUND (Hz_Tip - ArrowLen * CosArrow * CosAng + ArrowLen * SinArrow * SinAng),
                     IROUND (Vt_Tip - ArrowLen * CosArrow * SinAng - ArrowLen * SinArrow * CosAng),
                     IROUND (Hz_Tip), IROUND (Vt_Tip) );
           Pl_Line ( IROUND (Hz_Tip), IROUND (Vt_Tip),
                  IROUND (Hz_Tip - ArrowLen * CosArrow * CosAng - ArrowLen * SinArrow * SinAng),
                  IROUND (Vt_Tip - ArrowLen * CosArrow * SinAng + ArrowLen * SinArrow * CosAng) );
         END; {Pl_DrawArrow}


       PROCEDURE Pl_CentreBar (Hz_st, Vt_st, Len, Ang: Single);

         PROCEDURE DoubleBar;
           CONST
             BarWidth = ROUND (0.3 * Pl_1cm / 10);   {0.3 mm on plotter}
           VAR
             SinAng, CosAng : Single;
             Hz_Len, Vt_Len : Single;
             Cnr            : ARRAY [1..2] OF PlSizes;
           BEGIN
             Len      := ABS(Len);
             CosAng   := COS(Ang);
             SinAng   := SIN(Ang);
             Hz_Len   := Len * CosAng;
             Vt_Len   := Len * SinAng;
             Cnr[1,1] := IROUND (Hz_st + Hz_Len/2 - (BarWidth * SinAng) );
             Cnr[1,2] := IROUND (Vt_st + Vt_Len/2 + (BarWidth * CosAng) );
             Cnr[2,1] := IROUND (Hz_st + Hz_Len/2 + (BarWidth * SinAng) );
             Cnr[2,2] := IROUND (Vt_st + Vt_Len/2 - (BarWidth * CosAng) );
             Pl_Line (Cnr[1,1], Cnr[1,2], IROUND (Cnr[1,1] - Hz_Len), IROUND (Cnr[1,2] - Vt_Len) );
             Pl_Line ( IROUND (Cnr[2,1] - Hz_Len), IROUND (Cnr[2,2] - Vt_Len), Cnr[2,1], Cnr[2,2]);
           END; {DoubleBar}

         PROCEDURE SingleBar;
           VAR
             Hz_Len, Vt_Len: Single;
           BEGIN
             Hz_Len := (Len * COS(Ang));
             Vt_Len := (Len * SIN(Ang));
             Hz_St := Hz_St + Hz_Len/2;
             Vt_St := Vt_St + Vt_Len/2;
             Pl_Line ( IROUND(Hz_St), IROUND(Vt_St), IROUND (Hz_St - Hz_Len), IROUND (Vt_St - Vt_Len) );
           END; {SingleBar}

         BEGIN {Pl_CentreBar}
           Ang := -Ang;     {Angles defined here as: +ve CCW}
           IF (Len < 0)
              THEN SingleBar
              ELSE DoubleBar;
         END; {Pl_CentreBar}


       PROCEDURE Pl_DrawCrack (Hz_st, Vt_st, Hz_end, Vt_end: Single);
         CONST
           CrackWidth = ROUND (0.5 * Pl_1cm / 10);   {0.5 mm on plotter}
         VAR
           SineCrack, CosCrack : Single;
           Len, Hz_Len, Vt_Len : Single;
         BEGIN
           Hz_Len    := Hz_end - Hz_st;
           Vt_Len    := Vt_end - Vt_st;
           Len       := SQRT (SQR(Hz_Len) + SQR(Vt_Len));
           CosCrack  := CrackWidth * Hz_Len / Len;
           SineCrack := CrackWidth * Vt_Len / Len;
           Pl_MoveTo  (IROUND (Hz_st  - SineCrack), IROUND (Vt_st + CosCrack) );
           Pl_LineRel (IROUND (Hz_Len), IROUND (Vt_Len) );
           Pl_LineRel (IROUND (2 * SineCrack), IROUND (-2 * CosCrack) );
           Pl_LineRel (IROUND (-Hz_Len), IROUND (-Vt_Len) );
           Pl_LineRel (IROUND (-2 * SineCrack), IROUND (2 * CosCrack) );
         END; {Pl_DrawCrack}


    {* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *}


  END. {PlPrim}
