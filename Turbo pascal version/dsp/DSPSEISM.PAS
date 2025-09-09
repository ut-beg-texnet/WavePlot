  {Part of Dspl Unit}
   {Seismograms / 1-D graphing}
     CONST
       Sratio = 1.05;

     PROCEDURE SetVScale (Sc : ScaleTypes);
       VAR
         i, MxQ  : Integer;
         Mx      : Single;
       BEGIN
         GroupVScale := Sc;
         {Find vertical maximum and set vertical scales}
         CASE GroupVScale OF
            ScSetLink   : Mx := RoundPlaces (Sets.SLegMax[1], ScRound);
            ScLinked    : BEGIN
                            Mx  := 0;
                            FOR i := 1 TO QtySeism DO
                              IF (Seism[i].HstMap.Vmax > Mx)
                                 THEN Mx := Seism[i].HstMap.Vmax;
                            Mx := RoundPlaces (Mx, ScRound);
                          END; {ScLinked}
         END; {Case}
         CASE GroupVScale OF
            ScSet       : FOR i := 1 TO QtySeism DO
                            Seism[i].GR.LegMax := RoundPlaces (Sets.SLegMax[i], ScRound);
            ScNotLinked : FOR i := 1 TO QtySeism DO
                            Seism[i].GR.LegMax := RoundPlaces (Seism[i].HstMap.Vmax,ScRound);
            ScLinked,
              ScSetLink : FOR i := 1 TO QtySeism DO
                            Seism[i].GR.LegMax := Mx;
         END; {Case}
         CASE GroupVScale OF
            ScSet, ScNotLinked : BEGIN
                                   Mx  := 0;
                                   FOR i := 1 TO QtySeism DO
                                     IF (Seism[i].GR.LegMax > Mx)
                                        THEN Mx := Seism[i].GR.LegMax;
                                   Mx := RoundPlaces (Mx, ScRound);
                                 END; {ScLinked}
         END; {Case}
         {Correct for vertical scale settings}
         FOR i := 1 TO QtySeism DO
           IF (Seism[i].GR.LegMax = 0)
              THEN Seism[i].GR.LegMax := 1;
         FOR i := 1 TO QtySeism DO
           IF (  (Seism[i].HstMap.Vmax/Seism[i].GR.LegMax) > 1e4 )
              THEN Seism[i].GR.LegMax := RoundPlaces (Seism[i].HstMap.Vmax,ScRound);
       END; {SetVScale}

     PROCEDURE SetHScale (Sc : ScaleTypes);
       VAR
         i, MxQ  : Integer;
         Mx      : Single;
       BEGIN
         GroupHScale := Sc;
         {Find horizontal axis setting}
         CASE GroupHScale OF
           ScLinked:
                 BEGIN
                   MxQ := 0; Tmin := LargeSingle; Tmax := -LargeSingle;
                   FOR i := 1 TO QtySeism DO
                     WITH Seism[i].HstMap DO
                       BEGIN
                         IF (Tqty > MxQ) THEN MxQ := Tqty;
                         IF (Tstart < Tmin) THEN Tmin := Tstart;
                         IF (Time   > Tmax) THEN Tmax := Time;
                       END;
{                   MxQty    := MxQ;}
                 END;
           ScSetLink:
                 BEGIN
                   MxQ := 0;
                   FOR i := 1 TO QtySeism DO
                     WITH Seism[i].HstMap DO
                       IF (Tqty > MxQ) THEN MxQ := Tqty;
{                   MxQty := MxQ;}
                   Tmin  := Seis_HMin; Tmax := Seis_HMax;
                 END;
         END; {Case}
       END; {SetHScale}


     PROCEDURE FixLegMax;
       VAR
         i : Integer;
       BEGIN
         FOR i := 1 TO QtySeism DO
           Sets.sLegMax[i] := RoundPlaces ( (Seism[i].GR.LegMax / Vzoom), 2);
         IF (GroupHScale IN [ScLinked,ScSetLink])
            THEN BEGIN
                   Seis_HMin := RoundPlaces(HSst,2);
                   Seis_HMax := RoundPlaces(HSfin,2);
                   Qtyn := 1; Startn := 0;
                 END;
         Vzoom := 1;
         SetVScale (ScSet);
         IF (GroupHScale IN [ScLinked,ScSetLink]) THEN SetHScale (ScSetLink);
       END; {FixLegMax}


     PROCEDURE Ds_CalcQtys;
       VAR
         i             : Integer;
         TmpQty, TmpSt : Integer;
       BEGIN
         FOR i := 1 TO QtySeism DO
           WITH Seism[i] DO
           WITH HstMap DO
           WITH GR DO
             CASE (GroupHScale) OF
               ScLinked,
                 ScSetLink : BEGIN
                               HSst  := Tmin + (Tmax-Tmin)*Startn;
                               HSfin := HSst + (Tmax-Tmin)*Qtyn;
                               IF (ScalT=LogLogScal) THEN HSst:=0;
                               Hoff := (Tstart-HSst)/dt;
                               h0:=-ROUND(Hoff);
{                               IF (ScalT=LogLogScal) THEN h0:=0;}
                               Hoff:=0;
                               Qty := Tqty-h0;
                               IF (Tstart+(Tqty-1)*dt > HSfin)
                                  THEN Qty := TRUNC((HSfin-Tstart)/dt+1-h0);
                             END;
               ScNotLinked : BEGIN
                               h0  := ROUND (Tqty * Startn);
                               IF (ScalT=LogLogScal) THEN h0:=0;
                               Hoff := 0;
                               Qty := ROUND (Tqty * Qtyn);
                               IF (Qty>(Tqty-h0)) THEN Qty:=(Tqty-h0);
                               IF (Qty<0) THEN Qty:=0;
                             END;
             END; {Case}
       END; {Ds_CalcQtys}


     PROCEDURE Ds_InitSz;
       BEGIN
         Qtyn       := 1;
         Startn     := 0;
         Hzoom      := 1.25;
         Vzoom      := 1;
         LogDivV    := 4;
         LogDivH    := 2;
         ScRound    := 2;
         QtySWind   := QtySeism;
         SingleWind := False;        {Individual Windows for Seismograms}
         LineTypes  := False;
         SetVScale (ScNotLinked);    {Seismograms not vertically linked}
         SetHScale (ScNotLinked);    {Seismograms not horizontally linked}
         Ds_CalcQtys;
         Ds_CalcSeismWind;
       END; {Ds_InitSz}


     PROCEDURE Ds_ViewSeism (Choice : Char);
       VAR
         i : Integer;
       BEGIN
         WITH Seism[1].HstMap DO
           CASE Choice OF
             'Z'      : BEGIN
                          Qtyn := Qtyn / Hzoom;
                          IF (Qtyn < 0.01) THEN Qtyn := 0.01;
                        END;
             'z'      : BEGIN
                          Qtyn := Qtyn * Hzoom;
{                          IF (Qtyn>(1-Startn)) THEN Qtyn := 1-Startn;}
                        END;
(*             'X'      : Vzoom := Vzoom * 1.0204081632;      {eXpand vertical scale}
             'x'      : Vzoom := Vzoom * 0.98;*)
             'X'      : Vzoom := Vzoom * Hzoom;
             'x'      : Vzoom := Vzoom / Hzoom;
             'I'      : Hzoom := 1.00001 + (Hzoom-1) * 1.25;
             'i'      : Hzoom := 1.00001 + (Hzoom-1) * 0.8;
{             'I'      : Hzoom := SQR (Hzoom);
             'i'      : Hzoom := SQRT(Hzoom);}
             'C', 'c' : Sets.SameColour := NOT (Sets.SameColour);
             'D', 'd' : LineTypes  := NOT (LineTypes);      {Toggle Dashed Line Types}
             'F', 'f' : FixLegMax;  {Keep or Fix scales to current scale}
             'K', 'k' : FixLegMax;  {Keep or Fix scales to current scale}
             'H', 'h' : CASE GroupHScale OF
                          ScNotLinked : SetHScale (ScLinked);
                          ScLinked    : SetHScale (ScSetLink);
                          ScSetLink   : SetHScale (ScNotLinked);
                        END; {Case - change horizontal scale linkage}
             'V', 'v' : CASE GroupVScale OF
                          ScNotLinked : SetVScale (ScLinked);
                          ScLinked    : SetVScale (ScSet);
                          ScSet       : SetVScale (ScSetLink);
                          ScSetLink   : SetVScale (ScNotLinked);
                        END; {Case - change vertical scale linkage}
             'w', 'W' : IF (SingleWind)
                           THEN BEGIN
                                  SingleWind  := False;
                                  QtySWind    := QtySeism;
                                  Ds_CalcSeismWind;
                                  SetVScale (ScNotLinked);
                                  SetHScale (ScNotLinked);
                                END
                           ELSE BEGIN
                                  SingleWind := True;
                                  QtySWind    := 1;
                                  Ds_CalcSeismWind;
                                  SetVScale (ScLinked);
                                  SetHScale (ScLinked);
                                END;
             'N', 'n' : Ds_InitSz;
             '>'      : BEGIN IF (LogDivV < 10) THEN Inc (LogDivV);
                              IF (LogDivH < 10) THEN Inc (LogDivH); END;
             '<'      : BEGIN IF (LogDivV > 1)  THEN Dec (LogDivV);
                              IF (LogDivH > 1)  THEN Dec (LogDivH); END;
             '+'      : IF (ScRound < 4)
                           THEN BEGIN
                                  Inc (ScRound);
                                  SetVScale (GroupVScale);
                                  SetHScale (GroupHScale);
                                END;
             '-'      : IF (ScRound > 1)
                           THEN BEGIN
                                  Dec (ScRound);
                                  SetVScale (GroupVScale);
                                  SetHScale (GroupHScale);
                                END;
             '0'..'9' : CASE (Seism [Ival(Choice)].GR.ScalT) OF
                          LinScal  : Seism [Ival(Choice)].GR.ScalT := AbsLScal;
                          AbsLScal : Seism [Ival(Choice)].GR.ScalT := LogScal;
                          LogScal  : Seism [Ival(Choice)].GR.ScalT := LogLogScal;
                          LogLogScal   : Seism [Ival(Choice)].GR.ScalT := LinScal;
                        END;
             #0       : CASE ReadKey OF
                          #59         : Ds_ShowHelp(HelpHist);   {'F1' Help}
                          #71 {Home}  : Startn := 0;
                          #79 {End}   : Startn := (1 - Qtyn);
                          #75 {Left}  : BEGIN
                                          Startn := Startn - Qtyn*0.05;
{                                          IF (Startn < 0) THEN Startn := 0;}
                                        END;
                          #77 {Right} : BEGIN
                                          Startn := Startn + Qtyn*0.05;
{                                          IF (Startn > (1-Qtyn) ) THEN Startn := (1-Qtyn);}
                                        END;
                          #34         : BEGIN                            {Alt-G - graphics colours}
                                          IF (sets.colscale=Col_Grey)
                                             THEN sets.colscale:=Col_Blackb
                                             ELSE sets.colscale:=succ(sets.colscale);
                                          IF (Rainbow)
                                             THEN Ds_SetPalette
                                             ELSE Ds_ResetPalette;
                                        END;
                        END; {Case}
           END; {Case}
         Ds_CalcQtys;
       END; {Ds_ViewSeism}


     PROCEDURE Ds_GetSeisms;
       VAR
         i,j  : Integer;
         {Hnum0, Hmap:HstMap & GRs:GR, objects defined in Unit DspForm}
       BEGIN
         FOR i := 1 TO QtySeism DO
           WITH Seism[i] DO
             IF (FormSet)
                THEN BEGIN
                       Hmap := Seism[i].Hstmap;
                       GRs  := Seism[i].GR;
                       Hnum0 := Seism[i].Hnum;
                       DI_ReadHist (Hnum, Hist);
                       Parse (Hist, Form);
                       Hmap.Vmax := 0;
                       FOR j := 1 TO Hmap.Tqty DO
                         Hmap.Vmax := Max (Hmap.Vmax, ABS(Hist^[j]) );
{                       Hmap.HistVar    := FormName;   {Indicates a formula}
                       Seism[i].Hstmap := Hmap;
                       Seism[i].GR     := GRs;
                     END
                ELSE DI_ReadHist (Hnum, Hist);
       END; {Ds_GetSeisms}


     PROCEDURE Ds_SeismAxes;
       VAR
         i,j,ic      : Integer;
         p1,p2,p0,pn : Integer;
         Htick       : Integer;
       BEGIN
         Ds_SelectColour (DsCol_Axes);
         FOR i := 1 TO QtySWind DO
           BEGIN
             WITH SeismWind[i,2] DO
               BEGIN
                 Ds_SetWindow (SeismWind[i,2]);

                 {Horizontal Axis}
                   CASE Seism[i].GR.ScalT OF
                     LinScal   :  P2 := (Sz[2] DIV 2);
                     AbsLScal, LogScal,
                       LogLogScal  :  P2 := (Sz[2]-1);
                   END; {Case}
                   Line (0, P2, Sz[1]-1, P2);

                 {Horizontal TickMarks}
                   CASE Seism[i].GR.ScalT OF
                     LinScal, AbsLScal, LogScal : Htick:=10;
                     LogLogScal : Htick:=LogDivH;
                   END; {Case}
                   P0 := 0;
                   FOR j := 1 TO (Htick) DO
                     BEGIN
                       P1 := ROUND ((Sz[1]-1)*j/Htick);
                       IF (j <= Htick) THEN
                          CASE Seism[i].GR.ScalT OF
                            LinScal : Line (P1, P2+1, P1, P2-1);
                            AbsLScal, LogScal, LogLogScal : Line (P1, P2+2, P1, P2-2);
                          END; {Case}
                       IF (Seism[i].GR.ScalT IN [LogLogScal]) THEN
                          FOR ic := 1 TO 10 DO
                            BEGIN
                              PN := ROUND (P0+(P1-P0)*Ln(ic)/ln(10));
                              Line (PN, P2+1, PN, P2-1);
                            END; {For}
                       P0 := P1;
                     END; {For}

                 {Vertical Axis and TickMarks}
                   Line (0, 0, 0, Sz[2]-1);
                   ResetViewPort;
                   P1 := St[1];
                   IF (Seism[i].GR.ScalT IN [LinScal,AbsLScal])
                      THEN FOR j := 0 TO 4 DO
                             BEGIN
                               CASE Seism[i].GR.ScalT OF
                                 LinScal  :  P2 := ROUND (St[2]+(Sz[2]/2) - (j-2)/4*Sz[2]/Sratio);
                                 AbsLScal :  P2 := ROUND (St[2]+Sz[2]-1   -  j/4*Sz[2]/Sratio);
                               END; {Case}
                               Line (P1-1, P2, P1+1, P2);
                             END; {For j}

               END; {with SeismWind[i,2]}
           END; {For}
       END; {Ds_SeismAxes}


     PROCEDURE Ds_DrawSeism;
       VAR
         i  : Integer;

(*       PROCEDURE DrawGraph;
         VAR
           Hscale, Vscale : Single;
           h, dh          : Integer;
           hpos, hstp     : single;

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

         BEGIN {DrawGraph}
           WITH Seism[i] DO
           WITH HstMap DO
           WITH GR DO
           WITH SeismWind[i,2] DO
             BEGIN

               {Set vertical scaling}
                CASE ScalT OF
                  LinScal      : Vscale := -Sz[2]*0.5/Sratio/(LegMax/Vzoom);
                  AbsLScal     : Vscale := -Sz[2]/Sratio/(LegMax/Vzoom);
                  LogScal,
                    LogLogScal : Vscale := -Sz[2]/Sratio;
                END; {Case}
                CASE ScalT OF
                  LinScal                     : Ds_Voff :=  Sz[2] DIV 2;
                  AbsLScal,LogScal,LogLogScal : Ds_Voff :=  Sz[2]-1;
                END; {Case}
                IF (ScalT IN [LogScal,LogLogScal])
                   THEN BEGIN  LnVmax := LN(GR.LegMax/Vzoom); LnVmul := 1/(LogDivV*Ln10);  END;

               {Set horizontal scaling}
                CASE ScalT OF
                  LinScal,AbsLScal,LogScal:
                     IF (GroupHScale IN [ScLinked,ScSetLink])
                        THEN Hscale :=  Sz[1] * (Time-Tstart)/(Tmax-Tmin)/ ((Tqty-1)*Qtyn)
                        ELSE Hscale :=  Sz[1] / (Qty-1);
                  LogLogScal: BEGIN
                                Hscale := Sz[1];
                                IF (GroupHScale IN [ScLinked,ScSetLink])
                                   THEN LnHmax := LN(Tmax)
                                   ELSE LnHmax := LN(Time);
                                LnHmul := 1/(LogDivV*Ln10);
                              END;
                END; {Case}


               {Draw the graph}
               Ds_SetWindow (SeismWind[i,2]);
               dh := TRUNC (1/Hscale);
               IF (dh = 0) THEN dh := 1;
               hstp := dh*Hscale; h:=1+h0;
               hpos := Hoff*Hscale;
               MoveTo (Round(hpos), ROUND(Hist^[h]*Vscale)+Ds_Voff);
               REPEAT
                 h    := h+dh;
                 hpos := hpos+hstp;
                 LineTo ( ROUND (hpos), ROUND(Hist^[h]*Vscale)+Ds_Voff);
               UNTIL (h>=(h0+Qty));
             END; {With}

         END; {DrawGraph}
*)

       PROCEDURE Draw_LinLin;
         VAR
           Hscale, Vscale : Single;
           h, dh, hmax    : Integer;
           hpos, hstp     : single;
         BEGIN
           WITH Seism[i] DO
           WITH HstMap DO
           WITH GR DO
           WITH SeismWind[i,2] DO
             BEGIN
               Ds_SetWindow (SeismWind[i,2]);
               Vscale  := -Sz[2]*0.5 / Sratio / (GR.LegMax / Vzoom);
               Ds_Voff :=  Sz[2] DIV 2;
               IF (GroupHScale IN [ScLinked,ScSetLink])
                  THEN Hscale :=  Sz[1] * (Time-Tstart)/(Tmax-Tmin)/ ((Tqty-1)*Qtyn)
                  ELSE Hscale :=  Sz[1] / ((Tqty-1)*Qtyn);
               dh := TRUNC(1/Hscale);  IF (dh=0) THEN dh:=1;
               hpos := Hoff*Hscale; hstp := dh*Hscale; h:=1+h0;
               IF (h<=0) THEN BEGIN  hpos:=hpos+(1-h)*Hscale; h:=1;  END;
               IF (h>=Tqty) THEN BEGIN  hpos:=hpos+(1-h)*Hscale; h:=Tqty;  END;
               MoveTo (Round(hpos), ROUND(Hist^[h]*Vscale)+Ds_Voff);
               h    := h+dh;
               hmax := iMin(h0+Qty,Tqty);
               WHILE (h<=hmax) DO
                 BEGIN
                   hpos := hpos+hstp;
                   LineTo ( ROUND (hpos), ROUND(Hist^[h]*Vscale)+Ds_Voff);
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
           WITH SeismWind[i,2] DO
             BEGIN
               Ds_SetWindow (SeismWind[i,2]);
               Vscale  := -Sz[2] / Sratio / (GR.LegMax / Vzoom);
               Ds_Voff :=  Sz[2]-1;
               IF (GroupHScale IN [ScLinked,ScSetLink])
                  THEN Hscale :=  Sz[1] * (Time-Tstart)/(Tmax-Tmin)/ ((Tqty-1)*Qtyn)
                  ELSE Hscale :=  Sz[1] / ((Tqty-1)*Qtyn);
               dh := TRUNC(1/Hscale);  IF (dh=0) THEN dh:=1;
               hpos := Hoff*Hscale; hstp := dh*Hscale; h:=1+h0;
               IF (h<=0) THEN BEGIN  hpos:=hpos+(1-h)*Hscale; h:=1;  END;
               IF (h>=Tqty) THEN BEGIN  hpos:=hpos+(1-h)*Hscale; h:=Tqty;  END;
               MoveTo (Round(hpos), ROUND( ABS(Hist^[h])*Vscale)+Ds_Voff);
               hmax := iMin(h0+Qty,Tqty);
               WHILE (h<=hmax) DO
                 BEGIN
                   hpos := hpos+hstp;
                   LineTo ( ROUND (hpos), ROUND( ABS(Hist^[h])*Vscale)+Ds_Voff);
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
           WITH SeismWind[i,2] DO
             BEGIN
               Ds_SetWindow (SeismWind[i,2]);
               LnVmax := LN(GR.LegMax/Vzoom);
               LnVmul := 1/(LogDivV*Ln10);
               Vscale  := -Sz[2]/Sratio;
               Ds_Voff :=  Sz[2]-1;
               IF (GroupHScale IN [ScLinked,ScSetLink])
                  THEN Hscale :=  Sz[1] * (Time-Tstart)/(Tmax-Tmin)/ ((Tqty-1)*Qtyn)
                  ELSE Hscale :=  Sz[1] / ((Tqty-1)*Qtyn);
               dh := TRUNC(1/Hscale);  IF (dh=0) THEN dh:=1;
               hpos := Hoff*Hscale; hstp := dh*Hscale; h:=1+h0;
               IF (h<=0) THEN BEGIN  hpos:=hpos+(1-h)*Hscale; h:=1;  END;
               IF (h>=Tqty) THEN BEGIN  hpos:=hpos+(1-h)*Hscale; h:=Tqty;  END;
               MoveTo (ROUND(hpos), ROUND(LnVscale(Hist^[h]))+Ds_Voff);
               hmax := iMin(h0+Qty,Tqty);
               WHILE (h<=hmax) DO
                 BEGIN
                   hpos := hpos+hstp;
                   LineTo ( ROUND(hpos), ROUND(LnVscale(Hist^[h]))+Ds_Voff);
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
           WITH SeismWind[i,2] DO
             BEGIN
               LnVmax := LN(GR.LegMax/Vzoom);
               LnVmul := 1/(LogDivV*Ln10);
               Vscale  := -Sz[2]/Sratio;
               Ds_Voff :=  Sz[2]-1;

               Hscale := Sz[1];
               IF (GroupHScale IN [ScLinked,ScSetLink])
                  THEN LnHmax := LN(Tmax*Qtyn)
                  ELSE LnHmax := LN(Time*Qtyn);
               LnHmul := 1/(LogDivH*Ln10);

               Ds_SetWindow (SeismWind[i,2]);
               hpos := Tstart; hstp := Dt;
               MoveTo (0, ROUND(LnVscale(Hist^[1]))+Ds_Voff);
               FOR h := 2 TO Qty DO
                 BEGIN
                   hpos := hpos+hstp;
                   LineTo ( ROUND(LnHscale(hpos)),
                            ROUND(LnVscale(Hist^[h]))+Ds_Voff );
                 END;
             END; {With}
         END; {Draw_LogLog}

       PROCEDURE LegID;
         VAR
           s : string;
         BEGIN
           SetTextStyle (SmallFont, HorizDir, FontSz-1);
           Ds_SetWindow (SeismWind[i,3]);
           WITH Seism[i] DO
            WITH HstMap DO
             WITH SeismWind[i,3] DO
               BEGIN
                 SetTextJustify (LeftText, TopText);
                 s := '['+Strng(i,$0100)+'] ';
                 IF (formset) THEN s:=s+'f:';
                 s := s+VarName(HistVar)+''+VFuncName[HFunc];
                    {+ '(' + Strng (Hnum, $0100) + ')';}
                 IF (SingleWind)
                    THEN OutTextXY (0, i*14, s)
                    ELSE OutTextXY (0, 0,    s);
               END;
         END; {LegID}

       PROCEDURE ListSeism;
         VAR
           Divs : Word;
           s    : string;
         BEGIN
           Ds_SetWindow (LegWind);
           SetTextStyle (SmallFont, HorizDir, FontSz-1);
           SetTextJustify (LeftText, CenterText);
           Divs := 2*QtySeism + 10;
           IF (Divs < 20) THEN Divs := 20;

           MoveTo ( Hinset, TRUNC ( (7+2*i)/Divs * LegWind.Sz[2]) );
           WITH Seism[i].HstMap DO
             s := '['+Strng(i,$0100)+'] h'+Strng(Seism[i].Hnum,$0100)+' '
                 + VarName(HistVar) +VFuncName[HFunc]
                 + ' (' +Strng(Xp,$0100)+',' +Strng(Yp,$0100)+',' +Strng(Zp,$0100)+')';
           OutText (s);
           MoveTo ( Hinset, TRUNC ( (8+2*i)/Divs * LegWind.Sz[2]) );
           WITH Seism[i] DO
             IF (FormSet)
                THEN OutText (' F()=' + Form);
           ResetViewPort;
         END; {ListSeism}


       BEGIN {Ds_DrawSeism}
         Ds_SelectColour (DsCol_Seism);
         FOR i := 1 TO QtySeism DO
           BEGIN
             IF NOT (Sets.SameColour)
                THEN IF (i <> 8)
                        THEN Ds_SelectColour (i+8)
                        ELSE Ds_SelectColour (8);
             IF ( (LineTypes) AND (i IN [1,5,7,8,9..100]) )
                THEN SetLineType (i);
             CASE Seism[i].GR.ScalT OF
               LinScal  : Draw_LinLin;
               AbsLScal : Draw_ALinLin;
               LogScal  : Draw_LinLog;
               LogLogScal  : Draw_LogLog;
             END; {Case}
             LegID;
             ListSeism;
             ResetLineStyle;

             IF ( (LineTypes) AND NOT (i IN [1,5,7,8,9..100]) )
                THEN BEGIN
                       SetLineType (i);
                       SetColor (black);
                       CASE Seism[i].GR.ScalT OF
                         LinScal    : Draw_LinLin;
                         AbsLScal   : Draw_ALinLin;
                         LogScal    : Draw_LinLog;
                         LogLogScal : Draw_LogLog;
                       END; {Case}
                       ResetLineStyle;
                       Ds_SelectColour (DsCol_Seism);
                     END;
           END;
       END; {Ds_DrawSeism}


     PROCEDURE Ds_SeismLeg;
       VAR
         i                 : Integer;
         LegScale, LegNum  : Single;
         Digits            : Word;
         T_beg, T_end      : Single;
         N_beg, N_end      : Single;

       PROCEDURE WriteV_LinVal;
         BEGIN
           WITH Seism[i] DO
           WITH HstMap DO
             BEGIN
               Ds_SetWindow (SeismWind[i,1]);
               WITH SeismWind[i,1] DO
                 BEGIN
                   SetTextStyle (SmallFont, HorizDir, FontSz-1);
                   SetTextJustify (RightText, TopText);
                   OutTextXY (Sz[1]-1, -2, '+'+Strng(LegNum,Digits) );
{                   LegNum   := Roundplaces(LegMax/Vzoom,3);
                   OutTextXY (Sz[1]-1, -2, Strng(LegNum, $F301) );}
                   IF (Seism[i].GR.ScalT = LinScal)
                      THEN BEGIN
                             SetTextJustify (RightText, BottomText);
{                             OutTextXY (Sz[1]-1, Sz[2]-1, Strng(-LegNum,$F301) );}
                             OutTextXY (Sz[1]-1, Sz[2]-1, '-'+Strng(LegNum,Digits) );
                           END;
                 END;
             END; {With Seism[i]}
         END; {WriteV_LinVal}

       PROCEDURE WriteV_LinScal;
         BEGIN
           WITH Seism[i] DO
           WITH HstMap DO
             BEGIN
               Ds_SetWindow (SeismWind[i,1]);
               WITH SeismWind[i,1] DO
                 BEGIN
{                   SetTextStyle (SmallFont, HorizDir, FontSz-2);
                   SetTextJustify (RightText, TopText);
                   IF (LegScale > 0)
                      THEN OutTextXY (Sz[1]-8,  Sz[2] DIV 6, Strng(LegScale, $0000)+' '  )
                      ELSE OutTextXY (Sz[1]-8,  Sz[2] DIV 6, Strng(LegScale, $0000)  );
                   SetTextStyle (SmallFont, HorizDir, FontSz-1);
                   OutTextXY (Sz[1]-17, Sz[2] DIV 6, '  *10');}

                   SetTextStyle (SmallFont, HorizDir, FontSz-1);
                   SetTextJustify (RightText, TopText);
                   OutTextXY (Sz[1]-1,  Sz[2] DIV 6, ' *1E' +Strng(LegScale, $0000)+'');
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
               Ds_SetWindow (SeismWind[i,1]);
               WITH SeismWind[i,1] DO
                 BEGIN
                   {Vertical Axis Legends}
                   Stp := (Sz[2]-1)/Sratio / LogDivV;
                   SetTextJustify (LeftText, CenterText);
                   SetTextStyle (SmallFont, HorizDir, FontSz-1);
                   OutTextXY (0, (Sz[2]-1) - ROUND ((LogDivV-0.5)*Stp),
                              '*' + Strng(GR.LegMax / Vzoom, $F001 + Digits) );
                 END;
             END; {With Seism[i]}
         END; {WriteV_LogVal}

       PROCEDURE WriteV_LogScal;
         VAR
           Stp    : Single;
           j, P2  : Word;
         BEGIN
           WITH Seism[i] DO
           WITH HstMap DO
             BEGIN
               Ds_SetWindow (SeismWind[i,1]);
               WITH SeismWind[i,1] DO
                 BEGIN
                   Stp := (Sz[2]-1)/Sratio / LogDivV;
                   SetTextJustify (RightText, TopText);
                   FOR j := 1 TO LogDivV DO
                     BEGIN
                       P2 := (Sz[2]-1)-ROUND(j*Stp);
                       SetTextStyle (SmallFont, HorizDir, FontSz-2);
                       OutTextXY (Sz[1]-7, P2, Strng((j-LogDivV),$0000) );
                       SetTextStyle (SmallFont, HorizDir, FontSz-1);
                       IF (j = LogDivV)
                          THEN OutTextXY (Sz[1]-12, P2, '10')
                          ELSE OutTextXY (Sz[1]-17, P2, '10');
                       Line (Sz[1]-5, P2, Sz[1]-1, P2);
                     END; {For}
                 END;
             END; {With Seism[i]}
         END; {WriteV_LogScal}

       PROCEDURE WriteH_LogScal;
         VAR
           j, P1  : Word;
         BEGIN
           WITH Seism[i] DO
           WITH HstMap DO
             BEGIN
               Ds_SetWindow (SeismWind[i,2]);
               WITH SeismWind[i,2] DO
                 BEGIN
                   SetTextJustify (RightText, BottomText);
                   FOR j := 1 TO LogDivH DO
                     BEGIN
                       P1 := ROUND((Sz[1]-1)*j/LogDivH);
                       SetTextStyle (SmallFont, HorizDir, FontSz-2);
                       IF (j = LogDivH)
                          THEN OutTextXY (P1, Sz[2]-10, Strng((j-LogDivH),$0000) )
                          ELSE OutTextXY (P1+15, Sz[2]-10, Strng((j-LogDivH),$0000) );
                       SetTextStyle (SmallFont, HorizDir, FontSz-1);
                       IF (j = LogDivH)
                          THEN OutTextXY (P1-5, Sz[2]-6, '10')
                          ELSE OutTextXY (P1+5, Sz[2]-6, '10');
                     END; {For}
                 END;
             END; {With Seism[i]}
         END; {WriteH_LogScal}

       BEGIN {Ds_SeismLeg}
         IF (ScRound < 2)
            THEN Digits := 2
            ELSE Digits := ScRound;
         Digits := Digits SHL 8;

         SetTextStyle (SmallFont, HorizDir, FontSz-1);
         Ds_SelectColour (DsCol_Val);
         FOR i := 1 TO QtySWind DO
           WITH Seism[i] DO
           WITH HstMap DO
           WITH GR DO
             BEGIN
               {Horizontal Axis Legends}
(*                 IF (ScalT = LogLogScal)
                    THEN IF NOT (GroupHScale IN [ScLinked,ScSetLink])
                            THEN BEGIN  T_beg:=0;  T_end:=Time*Qtyn;  END
                            ELSE BEGIN  T_beg:=0;  T_end:=Tmax*Qtyn;  END
                    ELSE IF NOT (GroupHScale IN [ScLinked,ScSetLink])
{                            THEN BEGIN  T_beg:=Tstart+h0*Dt;  T_end:=Tstart+(h0+Qty-1)*Dt;  END}
                            THEN BEGIN  T_beg:=Tstart+h0*Dt;  T_end:=Tstart+(h0+(Tqty*Qtyn)-1)*Dt;  END
                            ELSE BEGIN  T_beg:=HSst;          T_end:=HSfin;  END;
*)

                 IF (GroupHScale IN [ScNotLinked])
                    THEN BEGIN
                           T_beg:=Tstart+h0*Dt;
                           T_end:=Tstart+(h0+(Tqty*Qtyn)-1)*Dt;
                           N_beg := 1+h0; N_end := h0+(Tqty*Qtyn);
                         END
                    ELSE BEGIN
                           T_beg:=HSst; T_end:=HSfin;
                           N_beg := 1+h0; N_end := h0+(Qty*Qtyn);
                         END;
                 IF (ScalT = LogLogScal)
                    THEN BEGIN  T_beg:=0; N_beg:=1;  END;

                 T_beg := Roundplaces(T_beg,4);
                 T_end := Roundplaces(T_end,4);
                 N_beg := Roundplaces(N_beg,4);
                 N_end := Roundplaces(N_end,4);

                 Ds_SetWindow (SeismWind[i,1]);
                 SetTextJustify (RightText, BottomText);
                 WITH SeismWind[i,1] DO
                   IF (ScalT <> LogLogScal) THEN
                      BEGIN
                        OutTextXY (Sz[1]-1, GR.Ds_Voff, Strng(T_beg, $F401) );
                        IF (ScalT = LinScal)
                           THEN OutTextXY (Sz[1]-1, Ds_Voff+10, ' (n=' + Strng(N_beg,$0100) + ')' )
                           ELSE OutTextXY (Sz[1]-1, Ds_Voff-10, ' (n=' + Strng(N_beg,$0100) + ')' );
{                       OutTextXY (Sz[1]-1, GR.Ds_Voff, Strng( (GR.St*Dt),$0201)+GR.xstr );}
                      END;

                 Ds_SetWindow (SeismWind[i,3]);
                 SetTextJustify (LeftText, BottomText);
                 WITH SeismWind[i,3] DO
                   BEGIN
                     OutTextXY (0, GR.Ds_Voff, '' + Strng(T_end,$F401) + xstr);
                     IF (ScalT = LinScal)
                        THEN OutTextXY (0, Ds_Voff+10, ' (n=' + Strng(N_end,$0200) + ')' )
                        ELSE OutTextXY (0, Ds_Voff-10, ' (n=' + Strng(N_end,$0200) + ')' );
                   END;

               {Vertical Axis Legends}
                 LegNum   := LegMax / Vzoom;
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
         Ds_SelectColour (DsCol_Scal);
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

       END; {Ds_SeismLeg}


     PROCEDURE Ds_SeismInfo;
       VAR
         i, Divs : Word;
         y, m, d, dow : Word;
       BEGIN
         Ds_SetWindow (LegWind);
         Ds_SelectColour (DsCol_List);
         Divs := 2*QtySeism + 10;
         IF (Divs < 20) THEN Divs := 20;
         WITH Seism[1] DO
         WITH HstMap DO
         WITH LegWind DO
           BEGIN
             SetTextJustify (LeftText, BottomText);
             SetTextStyle (SmallFont, HorizDir, FontSz+1);
             MoveTo ( 0, TRUNC (1.4/Divs * LegWind.Sz[2]) );
{             MoveTo ( Sz[1] DIV 2, TRUNC (1.0/Divs * LegWind.Sz[2]) );}
{             OutText ('WVPLOT v'+ver);}
             OutText ('   WVPLOT');
             SetTextStyle (SmallFont, HorizDir, FontSz-1);
             OutText ('(v'+ver+')');

             SetTextJustify (CenterText, CenterText);
             SetTextStyle (SmallFont, HorizDir, FontSz-1);
             MoveTo ( Sz[1] DIV 2, TRUNC (2.0/Divs * LegWind.Sz[2]) );
             GetDate(y,m,d,dow);
             OutText ('('+strng(m,0)+'/'+strng(d,0)+ '/'+strng(y,0)+')  ');

             SetTextStyle (SmallFont, HorizDir, FontSz);
             SetTextJustify (LeftText, CenterText);
             MoveTo ( Hinset, TRUNC (3.5/Divs * LegWind.Sz[2]) );
{             MoveTo ( Sz[1] DIV 2, TRUNC (3.3/Divs * LegWind.Sz[2]) );}
             OutText (' File = '+Sets.Fname);
             SetTextStyle (SmallFont, HorizDir, FontSz-1);
{             MoveTo ( Sz[1] DIV 2, TRUNC (4.4/Divs * LegWind.Sz[2]) );}
             MoveTo ( Hinset, TRUNC (4.4/Divs * LegWind.Sz[2]) );
             OutText ('  ('+Sets.Drive+':'+Sets.Dir+'\)');
             SetTextJustify (LeftText, CenterText);
             MoveTo ( Hinset, TRUNC (5.3/Divs * LegWind.Sz[2]) );
             OutText (' T-beg:  ' + Strng (Tstart,$F302) + GR.xstr );
             MoveTo ( Hinset, TRUNC (6.2/Divs * LegWind.Sz[2]) );
             OutText (' T-fin:  ' + Strng (Time,$F302) + GR.xstr );
             MoveTo ( Hinset, TRUNC (7.1/Divs * LegWind.Sz[2]) );
             OutText (' T-step: ' + Strng (dt, $F302) + GR.xstr );
           END; {With}
         ResetViewPort;
       END; {Ds_SeismInfo}


     PROCEDURE Ds_GraphSeism;
       BEGIN
         Ds_SelectColour (DsCol_Axes);
         Ds_DrawWindow (MsgWind);
         Ds_DrawWindow (LegWind);
         Ds_SeismAxes;
         Ds_DrawSeism;
         Ds_SeismLeg;
         Ds_SeismInfo;
         IF (Sets.ShowTitles) THEN Ds_Titles;
       END; {Ds_GraphSeism}

