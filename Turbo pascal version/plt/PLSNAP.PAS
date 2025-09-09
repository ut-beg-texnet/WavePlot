  {Part of Plot_Rsc Unit}
   {Plotting Snapshots / 2-D graphing, line-graphs and contours}


     PROCEDURE Pl_WriteLevel ( Pos  : Word;
                               Lev  : Single   );
       VAR
         Vpos, Hpos  : Word;
         LevMx       : Single;
         S           : String;
       BEGIN
         Pl_SetTextStyle (SmallFont, HorizDir, FontSz);
         Pl_SetTextJustify (CenterText, CenterText);
         LevMx := Max ( ABS(Level[Sets.QtyLevels]), ABS(Level[1]) );
         Vpos  := TRUNC ( Pl_LegWind.Sz[2] / 2 * (1 - Pos/(12+2) ) );
         Hpos  := Pl_LegWind.Sz[1] DIV 2;
         Pl_SetWindow (Pl_LegWind);
         Pl_MoveTo (Hpos, Vpos);
         S := Strng (Pos, $00);
         IF (Length(S) = 1)
            THEN Insert (' ', S, 1);
         Pl_OutText ( '['+ S + '] ' + Strng (Lev/LevMx,$F300) );
       END; {Pl_WriteLevel}


   { * * * Plotting of Snapshot Contours * * *}

     PROCEDURE Pl_Contours;
       VAR
         ContQty, C, l  : Word;
         Dy, Dx         : Word;
         xx, yy         : Word;
         Xfin, Yfin     : Word;
         Xstt, Ystt     : Word;
         HzOff, VtOff   : Integer;
         Xscale, Yscale : Single;
         cfact          : Single;
         pfact          : Single;
         ContourPt      : ARRAY [1..2] OF Point;
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

       PROCEDURE Pl_DrawCont (P1, P2 : Point);
         VAR
           Px1, Px2 : Pixels;
         BEGIN
           Px1[1] := HzOff + IROUND (Xscale * P1[X]);
           Px1[2] := VtOff + IROUND (Yscale * P1[Y]);
           Px2[1] := HzOff + IROUND (Xscale * P2[X]);
           Px2[2] := VtOff + IROUND (Yscale * P2[Y]);
           Pl_Line (Px1[1], Px1[2], Px2[1], Px2[2]);
         END; {Pl_DrawCont}


       BEGIN {Pl_Contours}
         Pl_SetLineType (1);

         WITH CurSnapRec DO
         WITH Scale2D DO
           BEGIN
             Xfin   := QuickX; Yfin   := QuickY;
             Xscale := Xstp * Pl_Xpx * Xqty / QuickX / Xlen;
             Yscale := Ystp * Pl_Ypx * Yqty / QuickY / Ylen;

             Xstt   := Xst + ROUND ( (Xqty/QuickX - 1)*0.5 * Xstp);
             Ystt   := Yst + ROUND ( (Yqty/QuickY - 1)*0.5 * Ystp);
             HzOff  := Pl_Xoff + (Xstt*Pl_Xpx) DIV Xlen;
             VtOff  := Pl_Yoff + (Ystt*Pl_Ypx) DIV Ylen;
           END;

         IF (Sets.QtyLevels < MaxCont)
            THEN ContQty := Sets.QtyLevels
            ELSE ContQty := MaxCont;
         cfact := Sets.QtyLevels / (ContQty+1);
         IF (ContQty <= (PenTot-1) )
            THEN pfact := ContQty / (ContQty+1)
            ELSE pfact := (PenTot-1) / (ContQty+1);
         FOR c := 1 TO ContQty DO
           BEGIN
             l := TRUNC (c*cfact + 1);
             Pl_SelectColour ( TRUNC(c*pfact+2) );
             Pl_SetWindow  (Pl_SnapWind);
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
                   BoxPt[4].Pt[X] := BoxPt[4].Pt[X] / 4;
                   BoxPt[4].Pt[Y] := BoxPt[4].Pt[Y] / 4;
                   BoxPt[4].Cval  := BoxPt[4].Cval  / 4;

                   IF ( TestContour ( 0, 1, Level[l], 1)AND
                        TestContour ( 0, 4, Level[l], 2) )
                           THEN Pl_DrawCont (ContourPt [1], ContourPt [2]);
                   IF ( TestContour ( 0, 1, Level[l], 1)AND
                        TestContour ( 1, 4, Level[l], 2) )
                           THEN Pl_DrawCont (ContourPt [1], ContourPt [2]);
                   IF ( TestContour ( 0, 4, Level[l], 1)AND
                        TestContour ( 1, 4, Level[l], 2) )
                           THEN Pl_DrawCont (ContourPt [1], ContourPt [2]);

                   IF ( TestContour ( 1, 3, Level[l], 1)AND
                        TestContour ( 1, 4, Level[l], 2) )
                           THEN Pl_DrawCont (ContourPt [1], ContourPt [2]);
                   IF ( TestContour ( 1, 3, Level[l], 1)AND
                        TestContour ( 4, 3, Level[l], 2) )
                           THEN Pl_DrawCont (ContourPt [1], ContourPt [2]);
                   IF ( TestContour ( 1, 4, Level[l], 1)AND
                        TestContour ( 4, 3, Level[l], 2) )
                           THEN Pl_DrawCont (ContourPt [1], ContourPt [2]);

                   IF ( TestContour ( 0, 2, Level[l], 1)AND
                        TestContour ( 0, 4, Level[l], 2) )
                           THEN Pl_DrawCont (ContourPt [1], ContourPt [2]);
                   IF ( TestContour ( 0, 2, Level[l], 1)AND
                        TestContour ( 2, 4, Level[l], 2) )
                           THEN Pl_DrawCont (ContourPt [1], ContourPt [2]);
                   IF ( TestContour ( 0, 4, Level[l], 1)AND
                        TestContour ( 2, 4, Level[l], 2) )
                           THEN Pl_DrawCont (ContourPt [1], ContourPt [2]);

                   IF ( TestContour ( 2, 3, Level[l], 1)AND
                        TestContour ( 2, 4, Level[l], 2) )
                           THEN Pl_DrawCont (ContourPt [1], ContourPt [2]);
                   IF ( TestContour ( 2, 3, Level[l], 1)AND
                        TestContour ( 4, 3, Level[l], 2) )
                           THEN Pl_DrawCont (ContourPt [1], ContourPt [2]);
                   IF ( TestContour ( 2, 4, Level[l], 1)AND
                        TestContour ( 4, 3, Level[l], 2) )
                           THEN Pl_DrawCont (ContourPt [1], ContourPt [2]);
                 END; {For Dx, Dy}
             Pl_WriteLevel (c, Level[l]);
           END; {Loop for each contour}
       END; {Pl_Contours}


   { * * * Plotting of Mult-Line Graphs * * *}

     PROCEDURE Pl_LineGr (Snp : Word);
       VAR
         xPix, yPix         : LongInt;
         ix, iy             : Integer;
         Pen, PenLev,MaxPen : Word;
         Lev                : Integer;
         Mx, Ang            : Single;
         CosLev, SinLev     : Integer;
         HzOff, VtOff       : Integer;
         Xstt, Ystt         : Word;
         Xblk, Yblk         : LongInt;
         P1, P2             : Pl_Point;

       BEGIN {Pl_LineGr}
         Pl_SetLineType (1);

         Pl_SetColor (1);
         WITH CurSnapRec DO
{         With SnapMap^ [CurSnap] DO}
         WITH Scale2D DO
           BEGIN
             Mx := Max ( ABS(MaxVal), ABS(MinVal) );
             Xblk  := (Xstp * Xqty * Pl_Xpx) DIV QuickX;
             Yblk  := (Ystp * Yqty * Pl_Ypx) DIV QuickY;
             Xstt  := Xst + ROUND ( (Xqty/QuickX - 1)/2 * Xstp);
             Ystt  := Yst + ROUND ( (Yqty/QuickY - 1)/2 * Ystp);
             HzOff := Pl_Xoff + (Xstt*Pl_Xpx) DIV Xlen;
             VtOff := Pl_Yoff + (Ystt*Pl_Ypx) DIV Ylen;
             IF (Sets.VectSet)
                THEN Pl_VectSc := Pl_1CM * Sets.VectSc
                ELSE Pl_VectSc := Xblk / Xlen * Vzoom / Mx / LineGzoom;
             IF NOT (Sets.LineCol)
                THEN MaxPen := 1
                ELSE IF (Sets.QtyLevels < PenTot)
                        THEN MaxPen := Sets.QtyLevels
                        ELSE MaxPen := PenTot-1;
             FOR Pen := 1 TO MaxPen DO
                 BEGIN
                   PenLev := (Pen * Sets.QtyLevels) DIV MaxPen;
                   Pl_SelectColour (Pen);
                   Pl_SetWindow  (Pl_SnapWind);

                   IF (Sets.Dirn IN [Vert, Both])
                      THEN FOR ix := 1 TO QuickX DO
                             BEGIN
                               xPix := HzOff + ( (ix-1)*Xblk) DIV Xlen;
                               FOR iy := 1 TO QuickY DO
                                 BEGIN
                                   yPix := VtOff + ( (iy-1)*Yblk) DIV Ylen;
                                   Lev := IROUND (SnapShot^[ix,iy] * Pl_VectSc);
                                   IF (Sets.Dirn = Vert)
                                      THEN BEGIN
                                             SinLev := 0;
                                             CosLev := Lev;
                                           END
                                      ELSE IF (Ax1 = 't')
                                              THEN BEGIN
                                                     SinLev := Lev;
                                                     CosLev := Lev;
                                                   END
                                              ELSE BEGIN
                                                     ang := Arctan0 ( ABS(QuickX/2 - ix), ABS(QuickY/2 - iy) );
                                                     SinLev := IROUND (Lev * SIN(Ang) );
                                                     CosLev := IROUND (Lev * COS(Ang) );
                                                   END;
                                   IF (iy = 1)
                                      THEN BEGIN
                                             P1[1] := IROUND(xPix+CosLev);
                                             P1[2] := IROUND(yPix+SinLev);
                                           END
                                      ELSE BEGIN
                                             P2[1] := IROUND(xPix+CosLev);
                                             P2[2] := IROUND(yPix+SinLev);
                                             IF ( NOT (Sets.LineCol) OR
                                                ( (  ( (GetLevel(SnapShot^[ix,iy]) -1) * MaxPen  DIV Sets.QtyLevels )
                                                     + 1 )  =  Pen) )
                                                THEN BEGIN
                                                       IF (  (Pl_CP[1] <> P1[1]) OR (Pl_CP[2] <> P1[2])  )
                                                          THEN Pl_MoveTo (P1[1], P1[2]);
                                                       Pl_LineTo (P2[1], P2[2]);
                                                     END;
                                             P1 := P2;
                                           END; {If iy=1}
                                 END; {For iy}
                             END; {For ix}

                   IF (Sets.Dirn IN [Horiz, Both])
                      THEN FOR iy := 1 TO QuickY DO
                             BEGIN
                               yPix := VtOff + ( (iy-1)*Yblk) DIV Ylen;
                               FOR ix := 1 TO QuickX DO
                                 BEGIN
                                   xPix := HzOff + ( (ix-1)*Xblk) DIV Xlen;
                                   Lev := IROUND (SnapShot^[ix,iy] * Pl_VectSc);
                                   IF (Sets.Dirn = Horiz)
                                      THEN BEGIN
                                             SinLev := Lev;
                                             CosLev := 0;
                                           END
                                      ELSE IF (Ax1 = 't')
                                              THEN BEGIN
                                                     SinLev := Lev;
                                                     CosLev := Lev;
                                                   END
                                              ELSE BEGIN
                                                     ang := Arctan0 ( ABS(QuickX/2 - ix), ABS(QuickY/2 - iy) );
                                                     SinLev := IROUND (Lev * SIN(Ang) );
                                                     CosLev := IROUND (Lev * COS(Ang) );
                                                   END;
                                   IF (ix = 1)
                                      THEN BEGIN
                                             P1[1] := IROUND(xPix+CosLev);
                                             P1[2] := IROUND(yPix+SinLev);
                                           END
                                      ELSE BEGIN
                                             P2[1] := IROUND(xPix+CosLev);
                                             P2[2] := IROUND(yPix+SinLev);
                                             IF ( NOT (Sets.LineCol) OR
                                                ( (  ( (GetLevel(SnapShot^[ix,iy]) -1) * MaxPen  DIV Sets.QtyLevels )
                                                     + 1 )  =  Pen) )
                                                THEN BEGIN
                                                       IF (  (Pl_CP[1] <> P1[1]) OR (Pl_CP[2] <> P1[2])  )
                                                          THEN Pl_MoveTo (P1[1], P1[2]);
                                                       Pl_LineTo (P2[1], P2[2]);
                                                     END;
                                             P1 := P2;
                                           END; {If ix=1}
                                 END; {For ix}
                             END; {For iy}

                   IF ( Sets.LineCol )
                      THEN Pl_WriteLevel (Pen, Level[PenLev]);
                 END; {For Pen}
           END; {With SnapMap^ and Scale2d}
       END; {Pl_LineGr}


   { * * * Plotting of Legend for Snapshot outputs * * * }

     PROCEDURE Pl_ListLevels;
       CONST
         Hinset = 200;
       VAR
         Lev, LevStep : Word;
         LevMx        : Single;
         S            : String;
         Mid          : Word;
         Vstep        : Single;
       BEGIN
         Pl_SetWindow (Pl_LegWind);
         Vstep := Pl_LegWind.Sz[2] / 2;
         Mid   := Pl_LegWind.Sz[1] DIV 2;
         Pl_SetTextStyle (SmallFont, HorizDir, FontSz);
         WITH CurSnapRec DO
           BEGIN
             Pl_SetTextJustify (LeftText, CenterText);
             Pl_SelectColour (PlPen_Axes);
             Pl_MoveTo ( Hinset, TRUNC(Vstep * (1+10.5/12))  );
             Pl_OutText ('  SNAP NUMBER ' + Strng (Snp,0) );
             Pl_MoveTo ( Hinset, TRUNC(Vstep * (1+9/12))  );
             Pl_OutText (' Var   = ' + VarName(SnapVar) );
             Pl_MoveTo ( Hinset, TRUNC(Vstep * (1+8/12))  );
             Pl_OutText (' Time  = ' + Strng (Time,$F300) );
             Pl_MoveTo ( Hinset, TRUNC(Vstep * (1+7/12))  );
             Pl_OutText (' Max   = ' + Strng (MaxVal,$F300) );
             Pl_MoveTo ( Hinset, TRUNC(Vstep * (1+6/12))  );
             Pl_OutText (' Min   = ' + Strng (MinVal,$F300) );
             IF (Sets.SnapSet = MLine)
                THEN BEGIN
                       Pl_MoveTo ( Hinset, TRUNC(Vstep * (1+4.5/12))  );
                       Pl_OutText ('    Vector Scale ');
                       Pl_MoveTo ( Hinset, TRUNC(Vstep * (1+3.5/12)) );
                       Pl_OutText ('  1CM : ' + Strng ( Pl_1CM /Pl_VectSc, $F300 )  );
                     END;

             IF ( (Sets.SnapSet <> MLine) OR (Sets.LineCol) )
                THEN BEGIN
                       Pl_SetTextJustify (CenterText, CenterText);
                       Lev := 1;
                       LevStep := (Sets.QtyLevels-1) DIV 8 + 1;
                       LevMx   := Max ( ABS(Level[Sets.QtyLevels]), ABS(Level[1]) );
                       Pl_MoveTo ( Mid, TRUNC(Vstep * (1+1.5/12))  );
                       Pl_OutText ('  Colour Ratios');
                       Pl_MoveTo ( Mid, TRUNC(Vstep * (1+0.5/12))  );
                       Pl_OutText (' Scale: * '+ Strng (LevMx, $F300) );
                     END; {Colour Listing}
           END; {With}
         Pl_ResetViewPort;
       END; {Pl_ListLevels}


  PROCEDURE Pl_PlotGeom (G, gID : LongInt);
    VAR
      st1, st2, fin1, fin2 : Integer;
      i                    : Integer;
    BEGIN {Pl_PlotGeom}
      Pl_SetWindow (Pl_SnapWind);
      Pl_SelectColour (PlPen_Geom);
      Pl_SetTextJustify (CenterText, CenterText);
      WITH GeomMap^ [G] DO
      WITH Scale2D DO
        BEGIN
          Pl_SetLineType (2);
          FOR i := 1 TO Mattot DO
            BEGIN
              WITH Mats[i] DO
                {DrawRectangle - Dashed}
                Pl_RECTANGLE ( Pl_Xoff + (Pl_Xpx*(i1-X1+1)) DIV Xlen, Pl_Yoff + (Pl_Ypx*(j1-Y1+1)) DIV Ylen,
                               Pl_Xoff + (Pl_Xpx*(i2-X1+1)) DIV Xlen, Pl_Yoff + (Pl_Ypx*(j2-Y1+1)) DIV Ylen );
            END;
          Pl_SetLineType (2);
          FOR i := 1 TO SourceTot DO
            BEGIN
              WITH Sources[i] DO
                {DrawRectangle - Filled}
                Pl_RECTANGLE ( Pl_Xoff + (Pl_Xpx*(i1-X1+1)) DIV Xlen, Pl_Yoff + (Pl_Ypx*(j1-Y1+1)) DIV Ylen,
                               Pl_Xoff + (Pl_Xpx*(i2-X1+1)) DIV Xlen, Pl_Yoff + (Pl_Ypx*(j2-Y1+1)) DIV Ylen );
              WITH Sources[i] DO
                IF ( (i1=i2) AND (j1=j2) )
                   THEN Pl_OutTextXY (Pl_Xoff + (Pl_Xpx*(i1-X1+1)) DIV Xlen, Pl_Yoff + (Pl_Ypx*(j1-Y1+1)) DIV Ylen, '*' );
            END;
          FOR i := 1 TO StopeTot DO
            BEGIN
              IF (Stopes^[i].stype IN [0,1,2] )
                 THEN Pl_SetLineType (5)
                 ELSE Pl_SetLineType (1);
              WITH Stopes^[i] DO
                {DrawRectangle - Filled}
                Pl_RECTANGLE ( Pl_Xoff + (Pl_Xpx*(i1-X1+1)) DIV Xlen, Pl_Yoff + (Pl_Ypx*(j1-Y1+1)) DIV Ylen,
                               Pl_Xoff + (Pl_Xpx*(i2-X1+1)) DIV Xlen, Pl_Yoff + (Pl_Ypx*(j2-Y1+1)) DIV Ylen );
            END;
        END; {With GeomMap^[S]}

      Pl_ResetLineStyle;
    END; {Pl_PlotGeom}


   { * * * Plotting of Snapshots * * *}

     PROCEDURE PlotSnap (Snp : Word);
       VAR
         Query : Char;
       BEGIN
         Pl_InitPlot;
         Pl_SelectColour (PlPen_Axes);
         Pl_DrawWindow (Pl_LegWind);
         Pl_DrawWindow (Pl_XtrWind);
         Pl_Set2Dscale;
         CASE  Sets.SnapSet OF
           ContMap : Pl_Contours;
           Mline   : Pl_LineGr (Snp);
         END; {Case}
         Pl_ListLevels;
         WITH SnapMap^ [Snp] DO
           IF (Scale2D.TrueGscale)
              THEN Pl_PlotGeom (gnum, gID);
         Pl_Descript;
         Ds_WaitMessage ('Send Plot to         Plotter (Y/N)?', Query);
         IF (Query IN ['Y', 'y'])
            THEN Pl_EndPlot(1)
            ELSE Pl_EndPlot(0);
         Pl_NamePlot;
       END; {PlotSnap}


     PROCEDURE Pl_WriteText;
       VAR
         i      : Integer;
         sTable : Text;
       BEGIN {Pl_WriteText}
         ASSIGN (sTable , (Sets.Drive + ':' + Sets.Dir + '\' + Sets.FName + '.TBL') );
         REWRITE (sTable);
         WITH Seism[1] DO
           writeln (sTable, Hmap.Tqty);
         WITH Seism[1] DO
           FOR i := 1 TO Hmap.Tqty DO
             writeln (sTable, Hist^[i]);
         CLOSE (sTable);
       END; {Pl_WriteText}

