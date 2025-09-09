
  {Part of Plot_Rsc Unit}
   {Seismograms / 1-D graphing}

     CONST
       Sratio = 1.05;

     PROCEDURE Pl_SeismAxes;
       VAR
         i,j   : Integer;
         p1,p2 : LongInt;
         Htick : Integer;
       BEGIN
         Pl_SelectColour (PlPen_Axes);
         FOR i := 1 TO QtySWind DO
           BEGIN
             WITH Pl_SeismWind[i,2] DO
               BEGIN
                 Pl_SetWindow (Pl_SeismWind[i,2]);

                 {Horizontal Axis and TickMarks}
                   CASE Seism[i].GR.ScalT OF
                     LinScal   :  P2 := (Sz[2] DIV 2);
                     AbsLScal, LogScal,
                       LogLogScal  :  P2 := 0;
                   END; {Case}
                   Pl_Line (0, P2, Sz[1]-1, P2);

                   CASE Seism[i].GR.ScalT OF
                     LinScal, AbsLScal, LogScal : Htick:=10;
                     LogLogScal : Htick:=LogDivV;
                   END; {Case}
                   FOR j := 1 TO (Htick-1) DO
                     BEGIN
                       P1 := IROUND ((Sz[1]-1)*j/Htick);
                       Pl_Line (P1, P2+20, P1, P2-20);
                     END;

                 {Vertical Axis and TickMarks}
                   Pl_Line (0, 0, 0, Sz[2]-1);
                   Pl_ResetViewPort;
                   P1 := St[1];
                   IF (Seism[i].GR.ScalT IN [LinScal,AbsLScal])
                      THEN FOR j := 0 TO 4 DO
                             BEGIN
                               CASE Seism[i].GR.ScalT OF
                                 LinScal  :  P2 := IROUND (St[2]+(Sz[2]/2) + (j-2)/4*Sz[2]/Sratio);
                                 AbsLScal :  P2 := IROUND (St[2]           +  j/4*Sz[2]/Sratio);
                               END; {Case}
                               Pl_Line (P1-20, P2, P1+20, P2);
                             END; {For j}
               END;
           END; {For}
       END; {Pl_SeismAxes}


     PROCEDURE Pl_DrawSeism;
       VAR
         i  : Integer;

       PROCEDURE Draw_LinLin;
         VAR
           Hscale, Vscale : Single;
           h, dh, hmax    : Integer;
           hpos, hstp     : single;
         BEGIN
           WITH Seism[i] DO
           WITH HstMap DO
           WITH GR DO
           WITH Pl_SeismWind[i,2] DO
             BEGIN
               Pl_SetWindow (Pl_SeismWind[i,2]);
               Vscale  :=  Sz[2]*0.5 / Sratio / (GR.LegMax / Vzoom);
               Pl_Voff :=  Sz[2] DIV 2;
               IF (GroupHScale IN [ScLinked,ScSetLink])
                  THEN Hscale :=  Sz[1] * (Time-Tstart)/(Tmax-Tmin)/ ((Tqty-1)*Qtyn)
                  ELSE Hscale :=  Sz[1] / ((Tqty-1)*Qtyn);
               dh := TRUNC(1/Hscale);  IF (dh=0) THEN dh:=1;
               hpos := Hoff*Hscale; hstp := dh*Hscale; h:=1+h0;
               IF (h<=0) THEN BEGIN  hpos:=hpos+(1-h)*Hscale; h:=1;  END;
               IF (h>=Tqty) THEN BEGIN  hpos:=hpos+(1-h)*Hscale; h:=Tqty;  END;
               Pl_MoveTo (IROUND(hpos), IROUND(Hist^[h]*Vscale)+Pl_Voff);
               h    := h+dh;
               hmax := iMin(h0+Qty,Tqty);
               WHILE (h<=hmax) DO
                 BEGIN
                   hpos := hpos+hstp;
                   Pl_LineTo ( IROUND (hpos), IROUND(Hist^[h]*Vscale)+Pl_Voff);
                   h    := h+dh;
                 END; {While}
             END; {With}
         END; {Draw_LinLin}

       PROCEDURE Draw_ALinLin;
         VAR
           Hscale, Vscale : Single;
           h, dh, hmax    : Integer;
           hpos, hstp     : single;
         BEGIN
           WITH Seism[i] DO
           WITH HstMap DO
           WITH GR DO
           WITH Pl_SeismWind[i,2] DO
             BEGIN
               Pl_SetWindow (Pl_SeismWind[i,2]);
               Vscale  :=  Sz[2] / Sratio / (GR.LegMax / Vzoom);
               Pl_Voff :=  0;
               IF (GroupHScale IN [ScLinked,ScSetLink])
                  THEN Hscale :=  Sz[1] * (Time-Tstart)/(Tmax-Tmin)/ ((Tqty-1)*Qtyn)
                  ELSE Hscale :=  Sz[1] / ((Tqty-1)*Qtyn);
               dh := TRUNC(1/Hscale);  IF (dh=0) THEN dh:=1;
               hpos := Hoff*Hscale; hstp := dh*Hscale; h:=1+h0;
               IF (h<=0) THEN BEGIN  hpos:=hpos+(1-h)*Hscale; h:=1;  END;
               IF (h>=Tqty) THEN BEGIN  hpos:=hpos+(1-h)*Hscale; h:=Tqty;  END;
               Pl_MoveTo (IROUND(hpos), IROUND( ABS(Hist^[h])*Vscale)+Pl_Voff);
               hmax := iMin(h0+Qty,Tqty);
               WHILE (h<=hmax) DO
                 BEGIN
                   hpos := hpos+hstp;
                   Pl_LineTo ( IROUND (hpos), IROUND( ABS(Hist^[h])*Vscale)+Pl_Voff);
                   h    := h+dh;
                 END; {While}
             END; {With}
         END; {Draw_ALinLin}

       PROCEDURE Draw_LinLog;
         VAR
           Hscale, Vscale : single;
           h, dh, hmax    : Integer;
           hpos, hstp     : single;
           LnVmax, LnVmul : Single;

         FUNCTION LnVscale(Yval:Single) : Single;
           VAR
             LnVal : Single;
           BEGIN
             IF (Yval = 0)
                THEN LnVal := 0
                ELSE BEGIN
                       LnVal := 1 - LnVmul * (LnVmax-LN(ABS(Yval)));
                       IF (LnVal<0) THEN LnVal:=0;
                     END;
             LnVscale := LnVal*Vscale;
           END; {LnVscale}

         BEGIN {Draw_LinLog}
           WITH Seism[i] DO
           WITH HstMap DO
           WITH GR DO
           WITH Pl_SeismWind[i,2] DO
             BEGIN
               Pl_SetWindow (Pl_SeismWind[i,2]);
               LnVmax := LN(GR.LegMax/Vzoom);
               LnVmul := 1/(LogDivV*Ln10);
               Vscale  :=  Sz[2]/Sratio;
               Pl_Voff :=  0;
               IF (GroupHScale IN [ScLinked,ScSetLink])
                  THEN Hscale :=  Sz[1] * (Time-Tstart)/(Tmax-Tmin)/ ((Tqty-1)*Qtyn)
                  ELSE Hscale :=  Sz[1] / ((Tqty-1)*Qtyn);
               dh := TRUNC(1/Hscale);  IF (dh=0) THEN dh:=1;
               hpos := Hoff*Hscale; hstp := dh*Hscale; h:=1+h0;
               IF (h<=0) THEN BEGIN  hpos:=hpos+(1-h)*Hscale; h:=1;  END;
               IF (h>=Tqty) THEN BEGIN  hpos:=hpos+(1-h)*Hscale; h:=Tqty;  END;
               Pl_MoveTo (IROUND(hpos), IROUND(LnVscale(Hist^[h]))+Pl_Voff);
               hmax := iMin(h0+Qty,Tqty);
               WHILE (h<=hmax) DO
                 BEGIN
                   hpos := hpos+hstp;
                   Pl_LineTo ( IROUND(hpos), IROUND(LnVscale(Hist^[h]))+Pl_Voff);
                   h    := h+dh;
                 END; {While}
             END; {With}
         END; {Draw_LinLog}

       PROCEDURE Draw_LogLog;
         VAR
           Hscale, Vscale : single;
           h, dh          : Integer;
           hpos, hstp     : single;
           LnVmax, LnVmul : Single;
           LnHmax, LnHmul : Single;

         FUNCTION LnVscale(Yval:Single) : Single;
           VAR
             LnVal : Single;
           BEGIN
             IF (Yval = 0)
                THEN LnVal := 0
                ELSE BEGIN
                       LnVal := 1 - LnVmul * (LnVmax-LN(ABS(Yval)));
                       IF (LnVal<0) THEN LnVal:=0;
                     END;
             LnVscale := LnVal*Vscale;
           END; {LnVscale}

         FUNCTION LnHscale(Xval:Single) : Single;
           VAR
             LnVal : Single;
           BEGIN
             IF (Xval = 0)
                THEN LnVal:=0
                ELSE BEGIN
                       LnVal := 1 - LnHmul * (LnHmax-LN(ABS(Xval)));
                       IF (LnVal<0) THEN LnVal:=0;
                     END;
             LnHscale := LnVal*Hscale;
           END; {LnHscale}

         BEGIN {Draw_LogLog}
           WITH Seism[i] DO
           WITH HstMap DO
           WITH GR DO
           WITH Pl_SeismWind[i,2] DO
             BEGIN
               LnVmax := LN(GR.LegMax/Vzoom);
               LnVmul := 1/(LogDivV*Ln10);
               Vscale  :=  Sz[2]/Sratio;
               Pl_Voff :=  0;

               Hscale := Sz[1];
               IF (GroupHScale IN [ScLinked,ScSetLink])
                  THEN LnHmax := LN(Tmax*Qtyn)
                  ELSE LnHmax := LN(Time*Qtyn);
               LnHmul := 1/(LogDivV*Ln10);

               Pl_SetWindow (Pl_SeismWind[i,2]);
               hpos := Tstart; hstp := Dt;
               Pl_MoveTo (0, IROUND(LnVscale(Hist^[h]))+Pl_Voff);
               FOR h := 2 TO Qty DO
                 BEGIN
                   hpos := hpos+hstp;
                   Pl_LineTo ( IROUND(LnHscale(hpos)),
                            IROUND(LnVscale(Hist^[h]))+Pl_Voff );
                 END;
             END; {With}
         END; {Draw_LogLog}

       PROCEDURE LegID;
         VAR
           s : string;
         BEGIN
           Pl_SetTextStyle (SmallFont, HorizDir, FontSz-1);
           Pl_SetWindow (Pl_SeismWind[i,3]);
           WITH Seism[i] DO
            WITH HstMap DO
             WITH Pl_SeismWind[i,3] DO
               BEGIN
                 Pl_SetTextJustify (LeftText, TopText);
                 s := '['+Strng(i,$0100)+'] ';
                 IF (formset) THEN s:=s+'f:';
                 s := s+VarName(HistVar)+''+VFuncName[HFunc];
                    {+ '(' + Strng (Hnum, $0100) + ')';}
                 IF (SingleWind)
                    THEN Pl_OutTextXY (0, Sz[2]-i*280, s)
                    ELSE Pl_OutTextXY (0, Sz[2]-71,    s);
                 IF (LineTypes)
                    THEN IF (SingleWind)
                            THEN Pl_Line (60, Sz[2]-i*280+70, Sz[1]-60, Sz[2]-i*280+70)
                            ELSE Pl_Line (60, Sz[2]-1,    Sz[1]-60, Sz[2]-1);
               END;
         END; {LegID}


       PROCEDURE ListSeism;
         CONST
           Hinset = 200;
         VAR
           Divs : Word;
           s    : string;
         BEGIN
           Pl_SetWindow (Pl_LegWind);
           Pl_SetTextStyle (SmallFont, HorizDir, FontSz-1);
           Pl_SetTextJustify (LeftText, CenterText);
           Divs := 2*QtySeism + 6;
           IF (Divs < 20) THEN Divs := 20;

           Pl_MoveTo ( Hinset, TRUNC ( (Divs-5-2*i)/Divs * Pl_LegWind.Sz[2]) );
           WITH Seism[i].HstMap DO
             s := '['+Strng(i,$0100)+'] h'+Strng(Seism[i].Hnum,$0100)+' '
                 + VarName(HistVar) +VFuncName[HFunc]
                 + ' (' +Strng(Xp,$0100)+',' +Strng(Yp,$0100)+',' +Strng(Zp,$0100)+')';
           Pl_OutText (s);
           Pl_MoveTo ( Hinset, TRUNC ( (Divs-6-2*i)/Divs * Pl_LegWind.Sz[2]) );
           WITH Seism[i] DO
             IF (FormSet)
                THEN Pl_OutText ('  F( ) = ' + Form);
           Pl_ResetViewPort;
         END; {ListSeism}

       BEGIN {Pl_DrawSeism}
         Pl_SelectColour (PlPen_Seism);
         FOR i := 1 TO QtySeism DO
           BEGIN
             IF NOT (Sets.SameColour)
                THEN IF (i <> 8)
                        THEN Pl_SelectColour (i+1)
                        ELSE Pl_SelectColour (i+1);
             IF ( (LineTypes) AND (i IN [1,5,7,8,9..100]) )
                THEN Pl_SetLineType (i);
             CASE Seism[i].GR.ScalT OF
               LinScal  : Draw_LinLin;
               AbsLScal : Draw_ALinLin;
               LogScal  : Draw_LinLog;
               LogLogScal  : Draw_LogLog;
             END; {Case}
             LegID;
             ListSeism;
             Pl_ResetLineStyle;

             IF ( (LineTypes) AND NOT (i IN [1,5,7,8,9..100]) )
                THEN BEGIN
                       Pl_SetLineType (i);
                       Pl_SelectColour (PlPen_Seism);
                       CASE Seism[i].GR.ScalT OF
                         LinScal    : Draw_LinLin;
                         AbsLScal   : Draw_ALinLin;
                         LogScal    : Draw_LinLog;
                         LogLogScal : Draw_LogLog;
                       END; {Case}
                       Pl_ResetLineStyle;
                       Pl_SelectColour (PlPen_Seism);
                     END;
           END;
       END; {Pl_DrawSeism}


   PROCEDURE Pl_SeismLeg;
     VAR
       i                 : Integer;
       LegScale, LegNum  : Single;
       Digits            : Word;
       T_beg, T_end      : Single;

     PROCEDURE WriteV_LinVal;
       BEGIN
         WITH Seism[i] DO
         WITH HstMap DO
           BEGIN
             Pl_SetWindow (Pl_SeismWind[i,1]);
             WITH Pl_SeismWind[i,1] DO
               BEGIN
                 Pl_SetTextStyle (SmallFont, HorizDir, FontSz-1);
                 Pl_SetTextJustify (RightText, TopText);
                 Pl_OutTextXY (Sz[1]-1, Sz[2]-1, '+'+Strng(LegNum,Digits) );
                 IF (Seism[i].GR.ScalT = LinScal)
                    THEN BEGIN
                           Pl_SetTextJustify (RightText, BottomText);
                           Pl_OutTextXY (Sz[1]-1, 0, '-'+Strng(LegNum,Digits) );
                         END;
               END;
           END; {With Seism[i]}
       END; {WriteV_LinVal}

     PROCEDURE WriteV_LinScal;
       BEGIN
         WITH Seism[i] DO
         WITH HstMap DO
           BEGIN
             Pl_SetWindow (Pl_SeismWind[i,1]);
             WITH Pl_SeismWind[i,1] DO
               BEGIN
{                 Pl_SetTextStyle (SmallFont, HorizDir, FontSz-2);
                 Pl_SetTextJustify (RightText, TopText);
                 IF (LegScale > 0)
                    THEN Pl_OutTextXY (Sz[1]-80,  5*Sz[2] DIV 6, Strng(LegScale, $0000)+' '  )
                    ELSE Pl_OutTextXY (Sz[1]-80,  5*Sz[2] DIV 6, Strng(LegScale, $0000)  );
                 Pl_SetTextStyle (SmallFont, HorizDir, FontSz-1);
                 Pl_OutTextXY (Sz[1]-220, (5*Sz[2] DIV 6)-40, '  *10');}

                 Pl_SetTextStyle (SmallFont, HorizDir, FontSz-1);
                 Pl_SetTextJustify (RightText, TopText);
                 Pl_OutTextXY (Sz[1]-1,  5*Sz[2] DIV 6, ' *1E' +Strng(LegScale, $0000)+'');
               END;
           END; {With Seism[i]}
       END; {WriteV_LinScal}

     PROCEDURE WriteV_LogVal;
       VAR
         Stp    : Single;
         j      : Word;
       BEGIN
         WITH Seism[i] DO
         WITH HstMap DO
           BEGIN
             Pl_SetWindow (Pl_SeismWind[i,1]);
             WITH Pl_SeismWind[i,1] DO
               BEGIN
                 {Vertical Axis Legends}
                 Stp := (Sz[2]-1)/Sratio / LogDivV;
                 Pl_SetTextJustify (LeftText, CenterText);
                 Pl_SetTextStyle (SmallFont, HorizDir, FontSz-1);
                 Pl_OutTextXY (0, IROUND ((LogDivV-0.5)*Stp),
                               '*' + Strng(GR.LegMax / Vzoom, $F001 + Digits) );
               END;
           END; {With Seism[i]}
       END; {WriteV_LogVal}

     PROCEDURE WriteV_LogScal;
       VAR
         Stp  : Single;
         j    : Word;
         P2   : LongInt;
       BEGIN
         WITH Seism[i] DO
         WITH HstMap DO
           BEGIN
             Pl_SetWindow (Pl_SeismWind[i,1]);
             WITH Pl_SeismWind[i,1] DO
               BEGIN
                 Stp := (Sz[2]-1)/Sratio / LogDivV;
                 Pl_SetTextJustify (RightText, TopText);
                 FOR j := 1 TO LogDivV DO
                   BEGIN
                     P2 := IROUND(j*Stp);
                     Pl_SetTextStyle (SmallFont, HorizDir, FontSz-2);
                     Pl_OutTextXY (Sz[1]-80, P2, Strng((j-LogDivV),$0000) );
                     Pl_SetTextStyle (SmallFont, HorizDir, FontSz-1);
                     IF (j = LogDivV)
                        THEN Pl_OutTextXY (Sz[1]-150, P2, '10')
                        ELSE OutTextXY (Sz[1]-230, P2, '10');
                     Pl_Line (Sz[1]-50, P2, Sz[1]-1, P2);
                   END; {For}
               END;
           END; {With Seism[i]}
       END; {WriteV_LogScal}

     PROCEDURE WriteH_LogScal;
       VAR
         j    : Word;
         P1   : LongInt;
       BEGIN
         WITH Seism[i] DO
         WITH HstMap DO
           BEGIN
             Pl_SetWindow (Pl_SeismWind[i,1]);
             WITH Pl_SeismWind[i,2] DO
               BEGIN
                 Pl_SetTextJustify (RightText, BottomText);
                 FOR j := 1 TO (LogDivV-1) DO
                   BEGIN
                     P1 := IROUND((Sz[1]-1)*j/LogDivV);
                     Pl_SetTextStyle (SmallFont, HorizDir, FontSz-2);
                     Pl_OutTextXY (P1+300, Sz[2]-200, Strng((j-LogDivV),$0000) );
                     Pl_SetTextStyle (SmallFont, HorizDir, FontSz-1);
                     IF (j = LogDivV)
                        THEN Pl_OutTextXY (P1+200, Sz[2]-120, '10')
                        ELSE Pl_OutTextXY (P1+100, Sz[2]-120, '10');
                   END; {For}
               END;
           END; {With Seism[i]}
       END; {WriteH_LogScal}


     BEGIN {Pl_SeismLeg}
       IF (ScRound < 2)
          THEN Digits := 2
          ELSE Digits := ScRound;
       Digits := Digits SHL 8;

       Pl_SetTextStyle (SmallFont, HorizDir, FontSz-1);
       Pl_SelectColour (PlPen_Val);
       FOR i := 1 TO QtySWind DO
         WITH Seism[i] DO
         WITH HstMap DO
         WITH GR DO
           BEGIN
             {Horizontal Axis Legends}
               IF (ScalT = LogLogScal)
                  THEN IF NOT (GroupHScale IN [ScLinked,ScSetLink] )
                          THEN BEGIN  T_beg:=0;  T_end:=Time;  END
                          ELSE BEGIN  T_beg:=0;  T_end:=Tmax;  END
                  ELSE IF NOT (GroupHScale IN [ScLinked,ScSetLink] )
{                            THEN BEGIN  T_beg:=Tstart+h0*Dt;  T_end:=Tstart+(h0+Qty-1)*Dt;  END}
                          THEN BEGIN  T_beg:=Tstart+h0*Dt;  T_end:=Tstart+(h0+(Tqty*Qtyn)-1)*Dt;  END
                          ELSE BEGIN  T_beg:=HSst;          T_end:=HSfin;  END;
               T_beg := Roundplaces(T_beg,3);
               T_end := Roundplaces(T_end,3);

               Pl_SetWindow (Pl_SeismWind[i,1]);
               Pl_SetTextJustify (RightText, BottomText);
               WITH Pl_SeismWind[i,1] DO
                 BEGIN
                   Pl_OutTextXY (Sz[1]-1, GR.Pl_Voff, Strng(T_beg, $F301) );
                 IF (ScalT = LinScal)
                    THEN Pl_OutTextXY (Sz[1]-1, Pl_Voff-200, ' (n=' + Strng( 1+h0,$0100) + ')' )
                    ELSE Pl_OutTextXY (Sz[1]-1, Pl_Voff+200, ' (n=' + Strng( 1+h0,$0100) + ')' );
{                   OutTextXY (Sz[1]-1, GR.Pl_Voff, Strng( (GR.St*Dt),$0201)+GR.xstr );}
               END;

               Pl_SetWindow (Pl_SeismWind[i,3]);
               Pl_SetTextJustify (LeftText, BottomText);
               WITH Pl_SeismWind[i,3] DO
                 BEGIN
{                   OutTextXY (0, GR.Pl_Voff, ' ' + Strng(T_end,$0201) );}
                   Pl_OutTextXY (0, GR.Pl_Voff, '' + Strng(T_end,$F301) + xstr);
                   IF (ScalT = LinScal)
                      THEN Pl_OutTextXY (0, Pl_Voff-200, ' (n=' + Strng( (h0+(Tqty*Qtyn)),$0200) + ')' )
                      ELSE Pl_OutTextXY (0, Pl_Voff+200, ' (n=' + Strng( (h0+(Tqty*Qtyn)),$0200) + ')' );
                 END;

           {Vertical Axis Legends}
             LegNum   := GR.LegMax / Vzoom;
             LegScale := LOG10 (LegNum);
             IF ( FRAC(LegScale) < 0)
                THEN LegScale := LegScale - 1;
             LegScale := INT (LegScale);
             LegNum := LegNum * EXP10(-TRUNC(LegScale));
             CASE ScalT OF
               LinScal, AbsLScal  : WriteV_LinVal;
               LogScal,LogLogScal : WriteV_LogVal;
             END; {Case}
           END; {With GR & HstMap}

       {Vertical Scales}
         Pl_SelectColour (PlPen_Scal);
         FOR i := 1 TO QtySWind DO
           WITH Seism[i] DO
           WITH HstMap DO
           WITH GR DO
             BEGIN
               LegNum   := LegMax / Vzoom;
               LegScale := LOG10 (LegNum);
               IF ( FRAC(LegScale) < 0)
                  THEN LegScale := LegScale - 1;
               LegScale := INT (LegScale);
               LegNum := LegNum * EXP10(-TRUNC(LegScale));
               CASE ScalT OF
                 LinScal, AbsLScal   : WriteV_LinScal;
                 LogScal, LogLogScal : WriteV_LogScal;
               END; {Case}
               IF (ScalT = LogLogScal)
                  THEN WriteH_LogScal;
             END; {With GR & HstMap}
     END; {Pl_SeismLeg}


   PROCEDURE Pl_SeismInfo;
     CONST
       Hinset = 200;
     VAR
       i, Divs : Word;
       y, m, d, dow : Word;
     BEGIN
       Pl_SetWindow (Pl_LegWind);
       Pl_SelectColour (PlPen_List);
       Divs := 2*QtySeism + 6;
       IF (Divs < 20) THEN Divs := 20;
       WITH Seism[1] DO
       WITH HstMap DO
       WITH LegWind DO
         BEGIN
           Pl_SetTextStyle (SmallFont, HorizDir, FontSz);
           Pl_SetTextJustify (LeftText, CenterText);
           Pl_MoveTo ( Hinset, TRUNC ((Divs-1.5)/Divs * Pl_LegWind.Sz[2]) );
           Pl_OutText (' File = '+Sets.Fname);
           Pl_SetTextStyle (SmallFont, HorizDir, FontSz-1);
           Pl_MoveTo ( Hinset, TRUNC ((Divs-2.4)/Divs * Pl_LegWind.Sz[2]) );
           Pl_OutText ('  ('+Sets.Drive+':'+Sets.Dir+'\)');

           Pl_SetTextJustify (LeftText, CenterText);
           Pl_MoveTo ( Hinset, TRUNC ((Divs-3.6)/Divs * Pl_LegWind.Sz[2]) );
           Pl_OutText (' T-beg:  ' + Strng (Tstart,$F302) + GR.xstr );
           Pl_MoveTo ( Hinset, TRUNC ((Divs-4.5)/Divs * Pl_LegWind.Sz[2]) );
           Pl_OutText (' T-fin:  ' + Strng (Time,$F302) + GR.xstr );
           Pl_MoveTo ( Hinset, TRUNC ((Divs-5.4)/Divs * Pl_LegWind.Sz[2]) );
           Pl_OutText (' T-step: ' + Strng (dt, $F302) + GR.xstr );
         END; {With}

       Pl_ResetViewPort;
     END; {Pl_SeismInfo}


   PROCEDURE Pl_PlotSeism;
     VAR
       Query : Char;
     BEGIN
       Pl_InitPlot;
       Pl_DrawWindow (Pl_GraphWind);
       Pl_DrawWindow (Pl_XtrWind);
       Pl_DrawWindow (Pl_LegWind);
       Pl_CalcSeismWind;
       Pl_SeismAxes;
       Pl_DrawSeism;
       Pl_SeismLeg;
       Pl_SeismInfo;
       Pl_Titles;
       Pl_Descript;
       Ds_WaitMessage ('Send Plot to         Plotter (Y/N)?', Query);
       IF (Query IN ['Y', 'y'])
          THEN Pl_EndPlot(1)
          ELSE Pl_EndPlot(0);
       Pl_NamePlot;
     END; {Pl_PlotSeism}

